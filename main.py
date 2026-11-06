# By Luis Andrew Madridijo (Implementation), Adrian Denzel Macayan (Benchmark)

import csv, os, re, statistics, time
from multiprocessing import Lock, Queue, Process, Value
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from rapidfuzz.distance import LCSseq

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ensure tesseract is in system PATH
os.environ["TESSDATA_PREFIX"] = os.path.dirname(os.path.abspath(__file__))
# one Tesseract thread per worker process to prevent the workers oversubscribing the CPU with OpenMP threads.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

BENCHMARK_WORKER_COUNTS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]
BENCHMARK_RUNS = 30
OPTIMAL_TOLERANCE = 0.05   # "optimal" = fewest workers within x% (set = 5%) of the best mean time

def load(image_dir, image_queue, n_workers):
    for name in sorted(os.listdir(image_dir)):
            if name.lower().endswith((".png")):
                 with Image.open(os.path.join(image_dir, name)) as img:
                    img.load()
                    image_queue.put((name, img))
    for _ in range(n_workers):
         image_queue.put(None)        # signal end of worker's work

def work(image_queue, output_lock, output_path, id_counter):
     while True:
        item = image_queue.get()
        if item is None:              # None means no more images in the queue
            break
        name, img = item

        if img.mode == "RGBA":        # flatten transparency onto white
            bg = Image.new("RGB", img.size, "white")
            bg.paste(img, mask=img.split()[3])
            img = bg

        img = ImageOps.grayscale(img)                                       # 1. grayscale
        img = ImageOps.autocontrast(img)                                    # 2. normalization
        img = img.filter(ImageFilter.MedianFilter(3))                       # 3. denoise

        start = time.perf_counter()
        text = pytesseract.image_to_string(img, lang="eng").strip()
        text = re.sub(r"[^A-Za-z]", "", text)                   # regex to keep letters only
        elapsed = time.perf_counter() - start

        with output_lock: # error safe shorthand for acquire() and release()
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                row_id = id_counter.value
                id_counter.value += 1
                csv.writer(f).writerow([row_id, name, text, f"{elapsed * 1000:.2f}"])   # processing time in ms

def read_column(csv_path, key_column, value_column):
    """
    Read a CSV into a dict of {key_column value: value_column value}.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row[key_column]: row[value_column].strip() for row in csv.DictReader(f)}


def error_check(labels_path, results_path, verbose=True):
    """
    Compare OCR output against the ground-truth labels, print the score, and return the recall % — TP/(TP+FN).
    Character recall = characters correctly recognized / total ground-truth characters.
    """
    truth_words = read_column(labels_path, "filename", "word")
    ocr_texts = read_column(results_path, "filename", "extracted_text")

    total_chars = 0       # every character in every ground-truth word
    matched_chars = 0     # how many of those words the OCR got right
    exact_matches = 0     # words where the OCR output is exactly the label

    for filename, truth_word in truth_words.items():
        # If an image has no OCR result, we treat it as empty so that all its characters count as missed.
        ocr_text = ocr_texts.get(filename, "")

        total_chars += len(truth_word)
        # Longest Common Subsequence length i.e. characters of truth_word found, in order, in the ocr_text.
        # Extra characters in ocr_text (like a stray "|") are ignored.
        matched_chars += LCSseq.similarity(truth_word, ocr_text)
        if ocr_text == truth_word:
            exact_matches += 1

    recall_percent = 100 * matched_chars / total_chars
    if verbose:
        print(f"Character recall: {matched_chars}/{total_chars} = {recall_percent:.2f}%")
        print(f"Exact word matches: {exact_matches}/{len(truth_words)}")
    return recall_percent


def run_pipeline(image_dir, n_workers, output_path):
    """
    Load -> Instantiate n_workers OCR workers -> Process CSV once -> Return the wall-clock time in seconds.
    """
    output_lock = Lock()
    image_queue = Queue()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["id", "filename", "extracted_text", "processing_time"])

    start = time.perf_counter()

    loader = Process(target=load, args=(image_dir, image_queue, n_workers))
    loader.start()

    id_counter = Value("i", 0)
    worker_list = []

    for _ in range(n_workers):
         worker = Process(target=work, args=(image_queue, output_lock, output_path, id_counter))
         worker_list.append(worker)
         worker.start()

    loader.join()

    for worker in worker_list:
         worker.join()

    return time.perf_counter() - start


def benchmark(image_dir, worker_counts=BENCHMARK_WORKER_COUNTS, runs=BENCHMARK_RUNS):
    """
    Time the OCR pipeline for some worker count.

    Metrics, each averaged over `runs` (set = 30) runs:
    Tn          mean wall-clock time with n workers (shown in ms)
    Speedup     S = T1 / Tn
    Efficiency  S / n
    Recall      character recall, to confirm accuracy does not change with n

    The runs are interleaved to lessen the impact of possible drift in the test machine load hitting worker count equally.
    We also run one untimed warm-up run to fill the file cache first.
    """
    counts = sorted(set(worker_counts) | {1})        # T1 to compute the base speed for the speedup formula
    scratch_path = "benchmark_tmp.csv"               # keep ocr_results.csv untouched
    labels_path = os.path.join(image_dir, "labels.txt")

    times = {n: [] for n in counts}
    recalls = {n: [] for n in counts}

    print(f"Benchmarking worker counts {counts} for {runs} runs each "f"on {os.cpu_count()} logical cores.")
    try:
        run_pipeline(image_dir, 1, scratch_path)     # warm-up for caching; not counted
        for i in range(1, runs + 1):
            for n in counts:
                times[n].append(run_pipeline(image_dir, n, scratch_path))
                recalls[n].append(error_check(labels_path, scratch_path, verbose=False))
            print(f"  run {i}/{runs} done")
    finally:
        if os.path.exists(scratch_path):
            os.remove(scratch_path)

    mean_time = {n: statistics.mean(times[n]) for n in counts}
    t1 = mean_time[1]

    print(f"\nBenchmark results (mean of {runs} runs per worker count)")
    print(f"{'Workers':>7} {'Mean Tn (ms)':>13} {'Speedup':>9} {'Efficiency':>11} {'Recall %':>9}")

    for n in counts:
        speedup = t1 / mean_time[n]
        print(f"{n:>7} {mean_time[n] * 1000:>13.1f} {speedup:>9.2f} {speedup / n:>10.0%} "f"{statistics.mean(recalls[n]):>9.2f}")

    best = min(mean_time.values())
    fastest = min(counts, key=lambda n: mean_time[n])
    optimal = min(n for n in counts if mean_time[n] <= best * (1 + OPTIMAL_TOLERANCE))

    print(f"\nS = T1 / Tn, with T1 = {t1 * 1000:.1f} ms (1 worker)")
    print(f"Fastest: {fastest} workers (S = {t1 / mean_time[fastest]:.2f}x)")
    print(f"Optimal: {optimal} workers (S = {t1 / mean_time[optimal]:.2f}x, "
          f"E = {t1 / mean_time[optimal] / optimal:.0%}) - "
          f"fewest workers within {OPTIMAL_TOLERANCE:.0%} of the best mean time")


if __name__ == "__main__":
    config = {}
    with open("config.txt") as f:
        for line in f:
            line = line.strip()
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()

    image_dir = config["image_dir"]
    n_workers = int(config["n_workers"])

    output_path = "ocr_results.csv"
    elapsed = run_pipeline(image_dir, n_workers, output_path)

    print("Base Pipeline Test")
    print(f"Time for {n_workers} workers: {elapsed:.4f} seconds")

    error_check(os.path.join(image_dir, "labels.txt"), output_path)

    print()
    benchmark(image_dir)
