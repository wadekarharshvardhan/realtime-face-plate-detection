class OCRProcessor:
    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['en'])  # You can add 'hi' for Hindi if needed

    def extract_text(self, image):
        import cv2
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # EasyOCR expects color images, so convert back if needed
        if len(gray.shape) == 2:
            img_for_ocr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            img_for_ocr = image

        results = self.reader.readtext(img_for_ocr, detail=0, paragraph=False)
        # Join all detected text parts
        return ' '.join(results).strip()

import re

# ...inside your main loop, after getting [next](http://_vscodecontentref_/3)...
# Try to extract a valid Indian plate pattern
##match = re.search(r'[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{1,4}', next.replace('\n', '').replace('\r', ''))
#