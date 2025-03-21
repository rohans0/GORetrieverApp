import sys, os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--task", type=str, default='cc')
parser.add_argument("--pro", type=str, default='2')
parser.add_argument("--file", type=str, default='./data/test.txt')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(args)

from PyQt6 import QtCore
from pyserini import search
from pyserini.search import SimpleSearcher
# from pyserini.search.lucene import LuceneSearcher
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5, torch
import nltk
from sentence_transformers import CrossEncoder
from transformers import T5ForConditionalGeneration
import torch.nn as nn
import torch.cuda as cuda

import json
import numpy as np

from tqdm import tqdm
from bs4 import BeautifulSoup as bs

# gui
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QBoxLayout,
    QCheckBox,
    QMessageBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QScrollArea,
    QLayoutItem 
)

# Main window
app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle("goretreiver")
window.resize(800, 600)
window.show()

# Scroll area
scroll_area = QScrollArea()
scroll_area.setWidgetResizable(True)
scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
window.setCentralWidget(scroll_area)

# Container widget
container_widget = QWidget()
scroll_area.setWidget(container_widget)

main_layout = QVBoxLayout()
container_widget.setLayout(main_layout)

# Create layout
input_layout = QVBoxLayout()
input_layout.setContentsMargins(0, 0, 0, 0)
input_layout.setSpacing(0)
main_layout.addLayout(input_layout)


tables_container = QVBoxLayout()
main_layout.addLayout(tables_container)


task_def = {
    'cc': 'Cellular Component',
    'bp': 'Biological Process',
    'mf': 'Molecular Function',
}

proid2name = np.load('./file/proid2name.npy', allow_pickle=True).item()
pmid2text = np.load('./file/pmid2text.npy', allow_pickle=True).item()

def get_input_data(file):
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    pro2text = {}
    with open(file) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.split('\t')
            pro = line[0]
            pmid = line[1].replace('\n', '')
            try:
                proname = proid2name[pro]
            except KeyError:
                print('Missing Protein Name', proname)
                continue

            if not pro2text.get(pro):
                pro2text[pro] = []

            text = pmid2text[pmid] 
            sentences = tokenizer.tokenize(text)
            # Discard information from documents containing only one sentence
            if len(sentences) == 1:
                continue
            if sentences[0] in pro2text[pro]:
                continue

            pro2text[pro].extend(sentences)




    # Table
    table = QTableWidget(len(pro2text), 2)
    table.setHorizontalHeaderLabels(['Protein ID', 'Tokenized Sentences'])

    table.setWordWrap(True)
    table.setEditTriggers(table.EditTrigger.NoEditTriggers)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    table.setMinimumHeight(600)

    row = 0
    for key, value in pro2text.items():
        table.setItem(row, 0, QTableWidgetItem(key))
        # table.setItem(row, 1, QTableWidgetItem(key))
        table.setItem(row, 1, QTableWidgetItem('\n'.join(value)))

        table.setRowHeight(row, 18*len(value))

        row += 1

    table.resizeColumnsToContents()

    
    global tables_container
    tables_container.addWidget(QLabel("Input data:"))
    tables_container.addWidget(table)
    table.lower()


    QApplication.processEvents()
    window.update()

    return pro2text


def data_extract(args):
    '''
    sentence extract
    return:
        pro2text[dictionary]:{
            'protein_id': extracted context(string)
        }
    '''
    pro2text = get_input_data(args.file)
    
    save_file = f'./test/{Path(args.file).stem}_{args.task}_dev_t5_texts.npy'
    print(save_file)
    if os.path.exists(save_file):
        data = np.load(save_file, allow_pickle=True).item()
        print('load from: ', save_file)
        print('data extract end!')
    else:
        # load models
        model = T5ForConditionalGeneration.from_pretrained('castorini/monot5-base-med-msmarco')
        model = nn.DataParallel(model)
        if cuda.is_available():
            model = model.cuda().module
        else:
            model = model.cpu().module

        T5tokenizer = MonoT5.get_tokenizer('t5-base')
        reranker = MonoT5(model, T5tokenizer)

        text_score = {}
        for pro in tqdm(pro2text): 
            query = f"what is the {task_def[args.task]} of protein {proid2name[pro]}?"
            sentences = pro2text[pro]
            if len(sentences) < 3:
                continue
            texts = []
            for sentence in sentences:
                texts.append(Text(sentence, {}, 0))
            scores = reranker.rerank(Query(query), texts)
            reranked = {}
            for i in range(len(scores)):
                reranked[sentences[i]] = float(scores[i].score)
            text_score[pro] = reranked

            reranked = sorted(reranked, key=lambda x:reranked[x], reverse=True)
            res = reranked[:len(reranked)//2]
            pro2text[pro] = ' '.join(res)

        np.save(save_file.replace('texts', 'texts_scores'), text_score)
        np.save(save_file, pro2text)
        print('data extract end!')

        data = pro2text



    # Table
    # table = QTableWidget(len(data), 2)
    # table.setWordWrap(True)
    # table.setEditTriggers(table.EditTrigger.NoEditTriggers)
    # table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    # table.setHorizontalHeaderLabels(['Protein ID', 'Tokenized Sentences'])
    # table.setMinimumHeight(400)
    #
    # row = 0
    # for key, value in data.items():
    #     table.setItem(row, 0, QTableWidgetItem(key))
    #     # table.setItem(row, 1, QTableWidgetItem(key))
    #     table.setItem(row, 1, QTableWidgetItem(', '.join(value)))
    #     print(','.join(value))
    #
    #     table.setRowHeight(row, 10*len(value))
    #
    #     row += 1
    #
    # table.resizeColumnsToContents()
    #
    # tables_container.addWidget(QLabel("Input data:"))
    # tables_container.addWidget(table)
    #
    # QApplication.processEvents()
    # window.update()
    #
    return data


def all_retrieval_dict(args):

    # make or get numpy cached file of current step
    save_file = f'./test/{Path(args.file).stem}_{args.task}_retrieval_all.npy'
    if os.path.exists(save_file):
        data = np.load(save_file, allow_pickle=True).item()
        print(save_file)
        print("All retrieval end!") 

    else:
        searcher = SimpleSearcher('./pro_index/')
        pro2text = data_extract(args)
        pro2go = np.load(f'./file/{args.task}_pro2go.npy', allow_pickle=True).item()
        retrieval_dict = {}
        for proid in tqdm(pro2text):
            k = 0
            res = []
            try:
                proname = json.loads(searcher.doc(proid).raw())['contents']
            except AttributeError:
                print('Missing Protein Name', proname)
                continue
            
            l = searcher.search(proname, 3000)
            for _ in l:
                if k > int(args.pro) + 2:
                    break
                _ = json.loads(_.raw)
                if _['id'] == proid:
                    continue
                else:
                    if pro2go.get(_['id']):
                        k += 1
                        res.append(pro2go[_['id']])
                    else:
                        res.append([])
            if k < int(args.pro) + 2:
                print(k, proid)
            retrieval_dict[proid] = res
        np.save(save_file, retrieval_dict)
        print('write:', save_file)
        print("All retrieval end!") 
        data = retrieval_dict      


    # Table
    # table = QTableWidget(len(data), 2)
    # table.setWordWrap(True)
    # table.setEditTriggers(table.EditTrigger.NoEditTriggers)
    # table.setHorizontalHeaderLabels(['Protein ID', 'Tokenized Sentences'])
    # table.setMinimumHeight(1000)
    #
    # row = 0
    # for key, value in data.items():
    #     table.setItem(row, 0, QTableWidgetItem(key))
    #     # table.setItem(row, 1, QTableWidgetItem(key))
    #     table.setItem(row, 1, QTableWidgetItem('\n '.join(value)))
    #
    #     # table.setRowHeight(row, 10*len(value))
    #
    #     row += 1
    #
    # table.resizeColumnsToContents()
    # table.resizeRowsToContents()
    #
    # layout.addWidget(QLabel("All retrieval dict:"))
    # layout.addWidget(table)
    #
    # QApplication.processEvents()
    # window.update()


    return data

def retrieval(args):
    # make or get numpy cached file of current step
    save_file = f'./test/{Path(args.file).stem}_{args.task}_retrieval.npy'
    if args.pro != '0':
        save_file = save_file.replace('retrieval', f'retrieval_pro_{args.pro}')

    if os.path.exists(save_file):
        data = np.load(save_file, allow_pickle=True).item()
        print(save_file)
        print("retrieval end!") 

    else:
        retrieval_dict = all_retrieval_dict(args)
        d = {}
        for proid in tqdm(retrieval_dict):
            k = 0
            res = []
            for l in retrieval_dict[proid]:
                if k==int(args.pro):
                    # print(k)
                    break
                if len(l) == 0:
                    continue
                else:
                    k += 1
                    res.extend(l)
            d[proid] = res

        np.save(save_file, d)
        print("write:", save_file)
        print("retrieval end!")

        data = d

    table = QTableWidget(len(data), 2)
    table.setHorizontalHeaderLabels(['Protein ID', 'GO IDs'])

    table.setWordWrap(True)
    table.setEditTriggers(table.EditTrigger.NoEditTriggers)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    table.setMinimumHeight(500)


    row = 0
    for key, value in data.items():
        table.setItem(row, 0, QTableWidgetItem(key))
        # table.setItem(row, 1, QTableWidgetItem(key))
        table.setItem(row, 1, QTableWidgetItem('\n '.join(value)))

        table.setRowHeight(row, 10*len(value))

        row += 1

    table.resizeColumnsToContents()

    tables_container.addWidget(QLabel("Retrieval dict:"))
    tables_container.addWidget(table)

    QApplication.processEvents()
    window.update()


    return data

def rerank(args):
    pro2text = data_extract(args)
    retrieval_data = retrieval(args)
    score = {}

    # make or get numpy cached file of current step
    score_dict = f'./test/{Path(args.file).stem}_{args.task}_t5_scores.npy'
    if os.path.exists(score_dict):
        print("score cache: ", score_dict)
        score = np.load(score_dict, allow_pickle=True).item()
    # if we have a cache , 

    goreader = SimpleSearcher('./go_index/')
    model = CrossEncoder(f'./cross_model/{args.task}_PubMedBERT_epoch1/', max_length=512)
    data = []
    

    for proid in tqdm(retrieval_data):
        predicts = []
        goids = []
        try:
            proname = proid2name[proid]
        except KeyError:
            continue
        try:
            query = f"The protein is \"{proname}\", the document is \"{pro2text[proid]}\"."
        except KeyError:
            print('text', proid)
            continue
        for goid in retrieval_data[proid]:
            if len(retrieval_data[proid]) == 0:
                continue
            if not score.get(proid):
                score[proid] = {}
            else:
                if score[proid].get(goid):
                    continue
            try:
                contents = json.loads(goreader.doc(goid.replace("GO:", '').replace("\n", "")).raw())['contents']
            except AttributeError:
                continue
            else:
                goids.append(goid)
                predicts.append([query, contents])
        if len(predicts) == 0:
            continue
        scores = model.predict(predicts,  batch_size = 96, show_progress_bar=False)
        for i in range(len(scores)):
            score[proid][goids[i]] = '%.3f'%float(scores[i])
            
    np.save(score_dict, score)  




    table = QTableWidget(2000, 3)
    table.setHorizontalHeaderLabels(['Protein ID', 'GO ID', 'Score'])

    table.setWordWrap(True)
    table.setEditTriggers(table.EditTrigger.NoEditTriggers)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    table.setMinimumHeight(1000)

    row = 0
    for proid in tqdm(retrieval_data):
        if len(retrieval_data[proid]) == 0:
            print("proid", proid)
            continue
        if not score.get(proid):
            print("proid", proid)
            continue           
        res = {}
        for goid in retrieval_data[proid]:
            res[goid] = str(score[proid].get(goid, 0))
        res = sorted(res.items(), key=lambda x: x[1], reverse=True)[:50]

        for item in res:
            data.append(proid + '\t' + item[0] + '\t' + item[1] + '\n')
            table.setItem(row, 0, QTableWidgetItem(proid))
            table.setItem(row, 1, QTableWidgetItem(item[0]))
            table.setItem(row, 2, QTableWidgetItem(item[1]))

            row += 1

    
    save_file = f'./result/{Path(args.file).stem}_{args.task}_t5_dev_rerank.txt'
    if args.pro != '0':
        save_file = save_file.replace('rerank', f'rerank_pro_{args.pro}')
    print(save_file)
    with open(save_file, 'w') as wf:
        wf.write(''.join(data))
    print('rerank end!')



    table.resizeColumnsToContents()
    table.resizeRowsToContents()

    tables_container.addWidget(QLabel("Output:"))
    tables_container.addWidget(table)

    QApplication.processEvents()
    window.update()


input_button : QPushButton
file_label : QLabel

def open_input_file():
    # while layout.count():
    #     child = layout.takeAt(0)
    #     if child.widget() and not isinstance(child.widget(), QPushButton):
    #         child.widget().deleteLater()
    args.file, _ = QFileDialog.getOpenFileName()
    if args.file:
        print(f"{args.file}")
        file_label.setText(f'Selected file: "{args.file}"')

def execute_button():
    if not args.file : return

    global tables_container
    tables_container = QVBoxLayout(container_widget)
    tables_container.setContentsMargins(0, 0, 0, 0)
    tables_container.setSpacing(0)

    main_layout.addLayout(tables_container)

    rerank(args)

def clear_cache():
    QMessageBox.information(
        window,
        "Cache Clear",
        "To clear the file cache, remove all files in ./test/",
        QMessageBox.StandardButton.Ok
    )

if __name__ == '__main__':

    input_btn = QPushButton("Open Input File")
    input_btn.clicked.connect(open_input_file)

    exec_btn = QPushButton("Execute File")
    exec_btn.clicked.connect(execute_button)

    clear_cache_btn= QPushButton("Clear Cached Files")
    clear_cache_btn.clicked.connect(clear_cache)

    file_label = QLabel(f'Selected file: "{args.file}"')

    mode_boxes_container = QWidget()
    mode_boxes_layout = QHBoxLayout()
    mode_boxes_container.setLayout(mode_boxes_layout)

    cc_box = QCheckBox(text='Cellular Component')
    cc_box.setCheckState(QtCore.Qt.CheckState.Checked)
    bp_box = QCheckBox(text='Biological Process')
    mf_box = QCheckBox(text='Molecular Function')

    def handle_cc(state):
        if state:
            bp_box.setCheckState(QtCore.Qt.CheckState.Unchecked)
            mf_box.setCheckState(QtCore.Qt.CheckState.Unchecked)
            args.task = 'cc'
            args.pro = 2

    def handle_bp(state):
        if state:
            cc_box.setCheckState(QtCore.Qt.CheckState.Unchecked)
            mf_box.setCheckState(QtCore.Qt.CheckState.Unchecked)
            args.task = 'bp'
            args.pro = 3

    def handle_mf(state):
        if state:
            cc_box.setCheckState(QtCore.Qt.CheckState.Unchecked)
            bp_box.setCheckState(QtCore.Qt.CheckState.Unchecked)
            args.task = 'mf'
            args.pro = 3

    cc_box.stateChanged.connect(handle_cc)
    bp_box.stateChanged.connect(handle_bp)
    mf_box.stateChanged.connect(handle_mf)


    mode_boxes_layout.addWidget(cc_box)
    mode_boxes_layout.addWidget(bp_box)
    mode_boxes_layout.addWidget(mf_box)

    input_layout.addWidget(clear_cache_btn)
    input_layout.addWidget(input_btn)
    if not cuda.is_available():
        input_layout.addWidget(QLabel(f'CUDA gpu not set up, instead using cpu.'))
    input_layout.addWidget(file_label)
    input_layout.addWidget(mode_boxes_container)
    input_layout.addWidget(exec_btn)

    input_layout.addStretch()

    # rerank(args)
    sys.exit(app.exec())

