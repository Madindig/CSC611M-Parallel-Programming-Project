import csv, os, re, time
from multiprocessing import Lock, Queue, Process, Value
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from rapidfuzz.distance import LCSseq

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = os.path.dirname(os.path.abspath(__file__))

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

        if img.mode == "RGBA":                       # flatten transparency onto white
            bg = Image.new("RGB", img.size, "white")
            bg.paste(img, mask=img.split()[3])
            img = bg

        img = ImageOps.grayscale(img)                                       # 1. grayscale
        img = ImageOps.autocontrast(img)                                    # 2. normalization
        img = img.filter(ImageFilter.MedianFilter(3))                       # 3. denoise

        start = time.perf_counter()
        text = pytesseract.image_to_string(img, lang="eng").strip()
        text = re.sub(r"[^A-Za-z]", "", text)         # keep letters only
        elapsed = time.perf_counter() - start

        with output_lock: # error safe shorthand for acquire() and release()
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                row_id = id_counter.value
                id_counter.value += 1
                csv.writer(f).writerow([row_id, name, text, f"{elapsed:.4f}"])

def read_column(csv_path, key_column, value_column):
    """Read a CSV into a dict of {key_column value: value_column value}."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row[key_column]: row[value_column].strip() for row in csv.DictReader(f)}


def error_check(labels_path, results_path):
    """Compare OCR output against the ground-truth labels and print the score.

    Character recall = characters correctly recognized / total ground-truth characters.
    """
    truth_words = read_column(labels_path, "filename", "word")
    ocr_texts = read_column(results_path, "filename", "extracted_text")

    total_chars = 0       # every character in every ground-truth word
    matched_chars = 0     # how many of those the OCR got right
    exact_matches = 0     # words where the OCR output is exactly the label

    for filename, truth_word in truth_words.items():
        # If an image has no OCR result, treat it as empty so all its characters count as missed.
        ocr_text = ocr_texts.get(filename, "")

        total_chars += len(truth_word)
        # Longest Common Subsequence length = characters of truth_word found, in order, in ocr_text.
        # Extra characters in ocr_text (like a stray "|") are ignored.
        matched_chars += LCSseq.similarity(truth_word, ocr_text)
        if ocr_text == truth_word:
            exact_matches += 1

    recall_percent = 100 * matched_chars / total_chars
    print(f"Character recall: {matched_chars}/{total_chars} = {recall_percent:.2f}%")
    print(f"Exact word matches: {exact_matches}/{len(truth_words)}")

if __name__ == "__main__":
    output_lock = Lock()
    image_queue = Queue()

    config = {}
    with open("config.txt") as f:
        for line in f:
            line = line.strip()
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()

    image_dir = config["image_dir"]
    n_workers = int(config["n_workers"])

    output_path = "ocr_results.csv"
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

    elapsed = time.perf_counter() - start

    print(f"Time for {n_workers} workers: {elapsed:.4f} seconds")

    error_check(os.path.join(image_dir, "labels.txt"), output_path)
