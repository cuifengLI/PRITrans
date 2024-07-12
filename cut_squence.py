import os
import sys
import pandas as pd
import openpyxl
# from excel_to_txt import mkdir
import numpy as np


##----以pos为中心，截取特定长度的AA----##
def cut_seq(Sequence, pos, PSA_length):
    if not Sequence:
        print("Sequence is empty")
    if int(pos) < 1 or int(pos) > len(Sequence):
        print("Position is out of the Sequence range")
    posQ = Sequence[:int(pos) - 1]  # 突变点前的序列部分
    posH = Sequence[int(pos):]  # 突变点后的序列部分
    var = Sequence[int(pos) - 1]  # 突变点上的碱基或氨基酸

    ##-------字符型变量使用前要先定义，后使用-----##
    posQN = ''
    posHN = ''
    SequenceN = ''
    tempQ = ''
    tempH = ''

    if (len(posQ) < int(0.5 * PSA_length) and len(posH) < int(0.5 * PSA_length)):  ## pos前后都小于二分之一的PSA要求长度
        tempQ = 'X' * (int(0.5 * PSA_length) - len(posQ))
        posQN = tempQ + posQ
        tempH = 'X' * (int(0.5 * PSA_length) - len(posH))
        posHN = posH + tempH

    elif (len(posQ) >= int(0.5 * PSA_length) and len(posH) < int(0.5 * PSA_length)):  ## pos前大于，后小于二分之一的PSA要求长度
        tempQ = posQ[-int(0.5 * PSA_length):]
        posQN = tempQ
        tempH = 'X' * (int(0.5 * PSA_length) - len(posH))
        posHN = posH + tempH

    elif (len(posQ) < int(0.5 * PSA_length) and len(posH) >= int(0.5 * PSA_length)):  ## pos后大于，前小于二分之一的PSA要求长度
        tempQ = 'X' * (int(0.5 * PSA_length) - len(posQ))
        posQN = tempQ + posQ
        tempH = posH[:int(0.5 * PSA_length)]
        posHN = tempH

    else:  ## pos前后都等于二分之一的PSA要求长度
        tempQ = posQ[-int(0.5 * PSA_length):]
        posQN = tempQ
        tempH = posH[:int(0.5 * PSA_length)]
        posHN = tempH

    SequenceN = posQN + var + posHN  ##  新的序列，用于返回
    return (SequenceN)


if __name__ == "__main__":

    # 打开需要裁剪序列的fasta文件，读取每一行；然后调用cut_seq函数截取序列，保存在一个新的txt文件中
    fasta_dir = "/Users/cuifengli/Desktop/protein_stability/datasets/S57_classify.fasta"

    seq_file = open(fasta_dir, "r")
    out_file = open("/Users/cuifengli/Desktop/protein_stability/Feature/ESM/S57_classify_181.fasta", "w")

    tou = []
    pos = []
    seq = []
    for lineID, line in enumerate(seq_file):
        if lineID == 0:
            tou.append(line)
            line = line.split('_')
            pos.append(line[3][1:-2])  # 从分割后的列表中取第四个元素（索引为3的元素）
        elif lineID % 2 == 1:
            line = line.replace('\n', '')
            seq.append(line)
            # print(len(seq[0]))
        elif lineID % 2 == 0:
            tou.append(line)
            line = line.split('_')
            pos.append(line[3][1:-2])

    cut_len = 180  # 设置裁剪长度为121，即突变点前后各60个氨基酸
    seq_cut = []
    for index, sequence in enumerate(seq):
        seq_cut.append(cut_seq(seq[index], pos[index], cut_len))
        # print(len(seq_cut[0]))

    for i in range(len(seq_cut)):
        out_file.write(tou[i])
        out_file.write(seq_cut[i] + '\n')

    seq_file.close()
    out_file.close()
