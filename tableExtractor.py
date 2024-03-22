import random
from PIL import Image, ImageEnhance
import statistics
import os
import string
from collections import Counter
from itertools import tee, count
from pytesseract import Output
from paddleocr import PaddleOCR
import layoutparser as lp
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
import torch

from convert2docx import add_table_to_docx

# Define yor output directory
output_directory = "output/output_csv"
ocr = PaddleOCR(use_angle_cls=True, lang='en',show_log=False) # need to run only once to download and load model into memory
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
modelTwo = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
def PIL_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv_to_PIL(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def pytess(cell_pil_img):
    cell_np_img = np.array(cell_pil_img)
    result_txt=[]
    results = ocr.ocr(cell_np_img, cls=False,det=True,rec=True)
    for result in results:
        if result != None:
            for line in result:
                if line != None:
                        #print(len(line))                            
                #print("text", line[1][0])
                    result_txt.append(line[1][0])
    return ' '.join(result_txt).strip()
   
        

def sharpen_image(pil_img):
    img = PIL_to_cv(pil_img)
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    pil_img = cv_to_PIL(sharpen)
    return pil_img

def uniquify(seq, suffs=count(1)):
    not_unique = [k for k, v in Counter(seq).items() if v > 1]
    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))
    for idx, s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            continue
        else:
            seq[idx] += suffix
    return seq

def binarizeBlur_image(pil_img):
    image = PIL_to_cv(pil_img)
    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]
    result = cv2.GaussianBlur(thresh, (5, 5), 0)
    result = 255 - result
    return cv_to_PIL(result)

def td_postprocess(pil_img):
    img = PIL_to_cv(pil_img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 100), (255, 5, 255))
    nzmask = cv2.inRange(hsv, (0, 0, 5), (255, 255, 255))
    nzmask = cv2.erode(nzmask, np.ones((3, 3)))
    mask = mask & nzmask
    new_img = img.copy()
    new_img[np.where(mask)] = 255
    return cv_to_PIL(new_img)

def table_detector(image, THRESHOLD_PROBA):
    feature_extractor = DetrFeatureExtractor(do_resize=True, size=800, max_size=800)
    encoding = feature_extractor(image, return_tensors="pt")
    #model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    with torch.no_grad():
        outputs = model(**encoding)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA
    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
    return (model, probas[keep], bboxes_scaled)

def table_struct_recog(image, THRESHOLD_PROBA):
    feature_extractor = DetrFeatureExtractor(do_resize=True, size=1000, max_size=1000)
    encoding = feature_extractor(image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = modelTwo(**encoding)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA
    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
    return (modelTwo, probas[keep], bboxes_scaled)

class TableExtractionPipeline:

    colors = ["red", "blue", "green", "yellow", "orange", "violet"]

    def add_padding(self, pil_img, top, right, bottom, left, color=(255, 255, 255)):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    def plot_results_detection(self, model, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        plt.imshow(pil_img)
        ax = plt.gca()
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cl = p.argmax()
            xmin, ymin, xmax, ymax = xmin - delta_xmin, ymin - delta_ymin, xmax + delta_xmax, ymax + delta_ymax
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=3))
            text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
            ax.text(xmin - 20, ymin - 50, text, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.show()

    def crop_tables(self, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        cropped_img_list = []
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            xmin, ymin, xmax, ymax = xmin - delta_xmin, ymin - delta_ymin, xmax + delta_xmax, ymax + delta_ymax
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cropped_img_list.append(cropped_img)
        return cropped_img_list

    def generate_structure(self, model, pil_img, prob, boxes, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
        plt.figure(figsize=(32, 20))
        #plt.imshow(pil_img)
        ax = plt.gca()
        rows = {}
        cols = {}
        idx = 0
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            xmin, ymin, xmax, ymax = xmin, ymin, xmax, ymax
            cl = p.argmax()
            class_text = model.config.id2label[cl.item()]
            text = f'{class_text}: {p[cl]:0.2f}'
            if (class_text == 'table row') or (class_text == 'table projected row header') or (class_text == 'table column'):
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=self.colors[cl.item()], linewidth=2))
                ax.text(xmin - 10, ymin - 10, text, fontsize=5, bbox=dict(facecolor='yellow', alpha=0.5))

            if class_text == 'table row':
                rows['table row.' + str(idx)] = (xmin, ymin - expand_rowcol_bbox_top, xmax, ymax + expand_rowcol_bbox_bottom)
            if class_text == 'table column':
                cols['table column.' + str(idx)] = (xmin, ymin - expand_rowcol_bbox_top, xmax, ymax + expand_rowcol_bbox_bottom)

            idx += 1

        plt.axis('on')
        #plt.show()
        return rows, cols

    def sort_table_featuresv2(self, rows: dict, cols: dict):
        rows_ = {table_feature: (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(rows.items(), key=lambda tup: tup[1][1])}
        cols_ = {table_feature: (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(cols.items(), key=lambda tup: tup[1][0])}
        return rows_, cols_

    def individual_table_featuresv2(self, pil_img, rows: dict, cols: dict):
        for k, v in rows.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            rows[k] = xmin, ymin, xmax, ymax, cropped_img

        for k, v in cols.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cols[k] = xmin, ymin, xmax, ymax, cropped_img

        return rows, cols

    def object_to_cellsv2(self, master_row: dict, cols: dict, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, padd_left):
        cells_img = {}
        header_idx = 0
        row_idx = 0
        previous_xmax_col = 0
        new_cols = {}
        new_master_row = {}
        previous_ymin_row = 0
        new_cols = cols
        new_master_row = master_row

        for k_row, v_row in new_master_row.items():
            _, _, _, _, row_img = v_row
            xmax, ymax = row_img.size
            xa, ya, xb, yb = 0, 0, 0, ymax
            row_img_list = []

            for idx, kv in enumerate(new_cols.items()):
                k_col, v_col = kv
                xmin_col, _, xmax_col, _, col_img = v_col
                xmin_col, xmax_col = xmin_col - padd_left - 10, xmax_col - padd_left
                xa = xmin_col
                xb = xmax_col

                if idx == 0:
                    xa = 0
                if idx == len(new_cols) - 1:
                    xb = xmax

                xa, ya, xb, yb = xa, ya, xb, yb
                #print(xa,ya,xb,yb)
                if xa < xb:
                    row_img_cropped = row_img.crop((xa, ya, xb, yb))
                    row_img_list.append(row_img_cropped)

            cells_img[k_row + '.' + str(row_idx)] = row_img_list
            row_idx += 1

        return cells_img, len(new_cols), len(new_master_row) - 1

    def clean_dataframe(self, df):
        for col in df.columns:
            df[col] = df[col].str.replace("'", '', regex=True)
            df[col] = df[col].str.replace('"', '', regex=True)
            df[col] = df[col].str.replace(':', '3', regex=True)
            df[col] = df[col].str.replace('€', '6', regex=True)
            df[col] = df[col].str.replace('-', '', regex=True)
        return df

    def convert_df(self, df):
        return df.to_csv().encode('utf-8')

    def create_dataframe(self, document,output_filename, cells_pytess_result: list, max_cols: int, max_rows: int):
        try:
            headers = cells_pytess_result[:max_cols]
            new_headers = uniquify(headers, (f' {x!s}' for x in string.ascii_lowercase))
            counter = 0
            cells_list = cells_pytess_result[max_cols:]
            df = pd.DataFrame("", index=range(0, max_rows), columns=new_headers)

            cell_idx = 0
            for nrows in range(max_rows):
                for ncols in range(max_cols):
                    df.iat[nrows, ncols] = str(cells_list[cell_idx])
                    cell_idx += 1

            for x, col in zip(string.ascii_lowercase, new_headers):
                if f' {x!s}' == col:
                    counter += 1

            header_char_count = [len(col) for col in new_headers]

            df = self.clean_dataframe(df)

            if (counter == len(new_headers)) or (statistics.median(header_char_count) < 6):
                #df.columns = uniquify(df.iloc[0], (f' {x!s}' for x in string.ascii_lowercase))
                df = df.iloc[1:, :]

            df.to_csv(os.path.join(output_directory, output_filename), index=False)
            add_table_to_docx(f"{output_directory}/{output_filename}", document)
        except IndexError as e:
            # Handle the IndexError gracefully (you can log it or print a message)
            print(f"Skipping create_dataframe function due to IndexError: {e}")

    def start_process(self, document, image_path: str, TD_THRESHOLD, TSR_THRESHOLD, padd_top, padd_left, padd_bottom, padd_right, delta_xmin, delta_ymin, delta_xmax, delta_ymax, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
        image = Image.open(image_path).convert("RGB")
        model, probas, bboxes_scaled = table_detector(image, THRESHOLD_PROBA=TD_THRESHOLD)

        if bboxes_scaled.nelement() == 0:
            print('No table found in the pdf-page image')
            return

        cropped_img_list = self.crop_tables(image, probas, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax)

        for i,unpadded_table in enumerate(cropped_img_list):
            table = self.add_padding(unpadded_table, padd_top, padd_right, padd_bottom, padd_left)
            table = binarizeBlur_image(table)
            table = sharpen_image(table)
            table = td_postprocess(table)

            model, probas, bboxes_scaled = table_struct_recog(table, THRESHOLD_PROBA=TSR_THRESHOLD)
            rows, cols = self.generate_structure(model, table, probas, bboxes_scaled, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom)
            rows, cols = self.sort_table_featuresv2(rows, cols)
            rows, cols = self.individual_table_featuresv2(table, rows, cols)
            cells_img, max_cols, max_rows = self.object_to_cellsv2(rows, cols, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, padd_left)

            cells_pytess_result = []
            for row_k, row_img_list in cells_img.items():
                for idx, cell in enumerate(row_img_list):
                    cell_pytess_result = pytess(cell)
                    cells_pytess_result.append(cell_pytess_result)

            filename = f"table_{i}.csv"
            self.create_dataframe(document, filename, cells_pytess_result, max_cols, max_rows)
            print(f'Table CSV saved as {filename}')
            
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# Define the path to the image you want to process
