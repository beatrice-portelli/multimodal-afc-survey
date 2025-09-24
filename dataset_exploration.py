"""
===============================================================================
Exploration and Reporting of Fact-Checking and Misinformation Datasets
===============================================================================

Purpose:
    This script loads and analyzes the following datasets, producing descriptive
    statistics, contingency tables and visualizations for:
      - label distributions
      - topic distributions
      - text length analyses (claims, evidence, captions, etc.)
      - saving reports as text, CSV and PDF figures

Datasets included (under `datasets/`):
    ├── FakeClaim/                # Fake vs. real video IDs
    ├── WarClaim/                 # Social media posts on the Israel–Ukraine war
    ├── FineFake/                 # Dataset with fine-grained labels
    ├── MOCHEG/                   # Claims and evidence with cleaned truth labels
    ├── Fakeddit datasetv2.0/     # Multimodal Reddit samples
    ├── fake-image-detection/     # (Fauxtography) Fake vs. real images
    ├── CovID_I/                  # COVID_I dataset
    ├── CovID_II/                 # COVID_II dataset
    ├── evons.csv                 # Single CSV file of evons data
    ├── Factify2/                 # Factify2 train/val sets
    └── ReCOVery/                 # ReCOVery fact-checking dataset

Structure:
    1. Setup (pandas display options, path imports, utility functions)
    2. One function per dataset:
         - fakeclaim()    ➔ FakeClaim
         - finefake()     ➔ FineFake
         - warclaim()     ➔ WarClaim
         - fauxtography() ➔ fake-image-detection
         - fakeddit()     ➔ Fakeddit datasetv2.0
         - mocheg()       ➔ MOCHEG
         - factify2()     ➔ Factify2
         - (placeholders for CovID_I, CovID_II, evons.csv, ReCOVery)
    3. Results are saved under `results/<DatasetName>/`

Requirements:
    • pandas
    • matplotlib
    • seaborn
    • utils.plt_save_image_pdf
    • src.paths for dataset file paths

Example:
    $ python dataset_statistics.py
"""


"""
 ├── FakeClaim/
 ├── WarClaim/
 ├── FineFake/
 ├── MOCHEG/
 ├── Fakeddit datasetv2.0/
 ├── fake-image-detection/    # Fauxtography
 ├── CovID_I/
 ├── CovID_II/
 ├── evons
 ├── Factify2/
 └── ReCOVery/
 """
 
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
import seaborn as sns
import numpy as np
import json
from utils import plt_save_image_pdf
 
def fakeclaim():
    df_fake = pd.read_csv("datasets/FakeClaim/Data/fake_video_ID.csv")
    df_real = pd.read_csv("datasets/FakeClaim/Data/real_video_ID.csv")

    dataset_name = "FakeClaim"

    print(f"{dataset_name} INFO")
    print(df_fake.columns)
    print(df_real.columns)

    print(df_fake.shape)
    print(df_real.shape)

    if not os.path.exists(f'results/{dataset_name}'):
        os.makedirs(f'results/{dataset_name}')

    with open(f'results/{dataset_name}/' + 'label.txt', 'w') as f:
        f.write(f"FAKE\n{df_fake.shape}\n\n")
        f.write(f"REAL\n{df_real.shape}\n")


def warclaim():
    dataset_name = 'warclaim'
    
    # read the json file with the data retrieved from youtube
    path = "datasets/WarClaim/war_youtube_titles.json"
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # delete the keys with the value "- YouTube", which are the error 
    for key in list(data.keys()):
        if data[key] == "- YouTube":
            del data[key]
    print(data.keys())

    length = 0
    n_char = 0
    max_char = 0
    min_char = 1000000
    for key in data.keys():
        length += 1
        n_char += len(data[key])
        if len(data[key]) > max_char:
            max_char = len(data[key])
        if len(data[key]) < min_char:
            min_char = len(data[key])

    if not os.path.exists(f'results/{dataset_name}'):
        os.makedirs(f'results/{dataset_name}')

    with open(f'results/{dataset_name}/' + 'fake.txt', 'w') as f:
        f.write(f"SHAPE: {len(data)} - {length}\n")
        f.write(f"MAX: {max_char}\n")
        f.write(f"MIN: {min_char}\n")
        f.write(f"MEAN: {n_char/length}\n")


def finefake():
    file_name = "datasets/FineFake/FineFake.pkl"
    df = pd.read_pickle(file_name)
    dataset_name = "FineFake"

    # print(df[df['text'].str.len() == df['text'].str.len().min()])

    print(f"{dataset_name} INFO")
    print(df.columns)
    print(df.shape)

    # print(df["label"].value_counts())
    print(df["fine-grained label"].value_counts())
    # print(df["topic"].value_counts())

    fig, axes = plt.subplots(1, 2, figsize=(5, 6))  # 1 row, 3 columns
    plt.suptitle("FineFake", fontsize=14, fontweight="bold")


    # List of labels for iteration
    labels = ["label", "fine-grained label"]
    for i, label in enumerate(labels):
        # Create the contingency table
        table = pd.crosstab(df["topic"], df[label])
        # Create heatmap on corresponding subplot
        if i == 0:
            sns.heatmap(table, cmap="Blues", fmt="d", linewidths=0.2, linecolor="gray", cbar=False, ax=axes[i])
        else:
            sns.heatmap(table, cmap="Blues", fmt="d", linewidths=0.2, linecolor="gray", cbar=True, ax=axes[i])
            axes[i].set_yticklabels([])
            axes[i].tick_params(left=False)
        axes[i].set_xlabel(label)
    # Adjust layout and save figure
    rect = plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color="black", linewidth=2, fill=False)
    fig.patches.append(rect)
    plt.tight_layout()
    plt_save_image_pdf(plt, f'results/FineFake/' + 'distribution_way')

    # create the crosstab
    table = pd.crosstab(df["topic"], df["label"])
    print(table)
    table = table.div(table.sum(axis=1), axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(5, 10))

    ax.imshow(table, cmap='Blues', interpolation='nearest', aspect='auto')
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xticks(range(len(table.columns)), table.columns)
    plt.tight_layout()


    # SAVE DATA
    if not os.path.exists(f'results/{dataset_name}'):
        os.makedirs(f'results/{dataset_name}')

    with open(f'results/{dataset_name}/' + '0_NOTE.txt', 'w') as f:
        pass

    with open(f'results/{dataset_name}/' + 'label_topic_distribution.txt', 'w') as f:
        f.write(f"LABELS\n{df['label'].value_counts()}\n\n")
        f.write(f"TOPICS\n{df['topic'].value_counts()}\n\n")
        for label in df["label"].unique():
            f.write(f"LABEL: {label}\n")
            f.write(f"MAX: {df[df['label'] == label]['text'].str.len().max()}\n")
            f.write(f"MIN: {df[df['label'] == label]['text'].str.len().min()}\n")
            f.write(f"MEAN: {df[df['label'] == label]['text'].str.len().mean()}\n\n")


    table.to_csv(f'results/{dataset_name}/' + 'topic_distribution.csv')
    plt_save_image_pdf(plt, f'results/{dataset_name}/' + 'topic_distribution')

    # create the crosstab with fine-grained label
    table = pd.crosstab(df["topic"], df["fine-grained label"])
    print(table)
    table = table.div(table.sum(axis=1), axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(5, 10))

    ax.imshow(table, cmap='Blues', interpolation='nearest', aspect='auto')
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xticks(range(len(table.columns)), table.columns)
    plt.tight_layout()

    with open(f'results/{dataset_name}/' + 'fine-grained label_topic_distribution.txt', 'w') as f:
        f.write(f"LABELS\n{df['fine-grained label'].value_counts()}\n\n")
        f.write(f"TOPICS\n{df['topic'].value_counts()}\n\n")
        for label in df["fine-grained label"].unique():
            f.write(f"LABEL: {label}\n")
            f.write(f"MAX: {df[df['fine-grained label'] == label]['text'].str.len().max()}\n")
            f.write(f"MIN: {df[df['fine-grained label'] == label]['text'].str.len().min()}\n")
            f.write(f"MEAN: {df[df['fine-grained label'] == label]['text'].str.len().mean()}\n\n")

    table.to_csv(f'results/{dataset_name}/' + 'fine-grained topic_distribution.csv')
    plt_save_image_pdf(plt, f'results/{dataset_name}/' + 'fine-grained topic_distribution')


def mocheg():
    df = pd.read_csv("results/MOCHEG/Corpus2_selected_claim_id.csv", sep=",")

    df_unique_claim_id = pd.DataFrame(columns=df.columns)
    dataset_name = "MOCHEG"
    os.makedirs(f'results/{dataset_name}', exist_ok=True) if not os.path.exists(f'results/{dataset_name}') else None
    print(f"{dataset_name} INFO")
    for set in ["train", "val", "test"]:
        #print(f"{set} INFO"  + "="*50)
        for file in ["Corpus2"]:# "img_evidence_qrels", "text_evidence_qrels_article_level"]: # "Corpus2", , "text_evidence_qrels_sentence_level"
            df = pd.read_csv("datasets/mocheg/" + set + "/" + file + ".csv", sep=",")
            #print(f"- {file}")
            #print(df.shape)
            # print(df.columns)
            # print unique values of the collum "claim_id"
            # create df_unique_claim_id void with collum of Corpus2
            
            # print(df_unique_claim_id.columns)
            #print(f"CLAIM_ID: {df['claim_id'].nunique()}")
            # for unique claim_id in df save first row in df_unique_claim_id
            for claim_id in df['claim_id'].unique():
                df_unique_claim_id = pd.concat([df_unique_claim_id, df[df['claim_id'] == claim_id].iloc[[0]]], ignore_index=True)
            print(df_unique_claim_id.shape)

    print(df_unique_claim_id.shape)
    with open(f'results/{dataset_name}/' + 'cleaned_truthfulness.txt', 'w') as f:
        for label in df_unique_claim_id["cleaned_truthfulness"].unique():
            f.write(f"LABEL: {label}\n")
            f.write(f"COUNT: {df_unique_claim_id[df_unique_claim_id['cleaned_truthfulness'] == label].shape[0]}\n")
            # mean, max and min length of the text in claim
            f.write(f"MAX: {df_unique_claim_id[df_unique_claim_id['cleaned_truthfulness'] == label]['Origin'].str.len().max()}\n")
            f.write(f"MIN: {df_unique_claim_id[df_unique_claim_id['cleaned_truthfulness'] == label]['Origin'].str.len().min()}\n")
            f.write(f"MEAN: {df_unique_claim_id[df_unique_claim_id['cleaned_truthfulness'] == label]['Origin'].str.len().mean()}\n\n")





            # print(df_unique_claim_id.columns)
        #     # print number of unique claim_id
        #     if file == "Corpus2":
        #         print(f"n CLAIM: {df['claim_id'].nunique()}")

        #     # # print number of unique evidence_id
        #     if file != "text_evidence_qrels_sentence_level":
        #         print(f"n EVI: {df['evidence_id'].nunique()}")

        #     # select a claim_id and save in a file all the rows with that claim_id
        #     # selected_claim_id = df['claim_id'].iloc[0]
        #     # df_selected = df[df['claim_id'] == selected_claim_id]
        #     # df_selected.to_csv(f"results/{dataset_name}/" + file + "_selected_claim_id.csv", index=False)
        #     # print(f"CLAIM_ID: {selected_claim_id}")
        #     # print(df_selected.shape)
        #     # # for each collum check the different values
        #     # for col in df_selected.columns:
        #     #     print(f"{col}: {df_selected[col].nunique()}")
            
        #     # create list selected_evidence_id that are evidence_id in df_selected
        #     selected_evidence_id = [0, 1, 2, 3] #df_selected['evidence_id'].unique()
        #     print(f"EVIDENCE_ID: {(selected_evidence_id)}")
            
        #     # WRONNNNGGFGG tO DOOOOOOO
        #     # for each evidence_id in selected_evidence_id save in a file all the rows with that evidence_id
        #     for selected_evidence_id in selected_evidence_id:
        #         df_selected = df[df['evidence_id'] == selected_evidence_id]
        #         df_selected.to_csv(f"results/{dataset_name}/" + file + "_selected_evidence_id_" + str(selected_evidence_id) + ".csv", index=False)
        #         print(f"EVIDENCE_ID: {selected_evidence_id}")
        #         print(df_selected.shape)
        #         # for each collum check the different values
        #         for col in df_selected.columns:
        #             print(f"{col}: {df_selected[col].nunique()}")

        #     # exit()


        
        # # index = 10
        # # for col in df.columns:
        # #     print(f"{col}: {df[col].iloc[index]}")

    # df_3 = pd.read_csv("datasets/mocheg/Corpus3.csv", sep=",")
    # print(f"Corpus3 INFO")
    # print(df_3.shape)
 
 
def fakeddit():
    # # read a csv file but use a progress bar
    df_train = pd.read_csv("datasets/Fakeddit datasetv2.0/multimodal_only_samples/multimodal_train.tsv", sep="\t")
    df_val = pd.read_csv("datasets/Fakeddit datasetv2.0/multimodal_only_samples/multimodal_validate.tsv", sep="\t")
    df_test = pd.read_csv("datasets/Fakeddit datasetv2.0/multimodal_only_samples/multimodal_test_public.tsv", sep="\t")
    # print INFO about the dataset
    dataset_name = "Fakeddit"
    print(f"{dataset_name} INFO")
    # print(df_train.columns)
    # print(df_val.columns)
    # print(df_test.columns)
    # print(df_train.shape)
    # print(df_val.shape)
    # print(df_test.shape)

    # # concat the three dataframes
    df = pd.concat([df_train, df_val, df_test])

    # # print the shape of the dataframe
    print(df.shape)
    # print(df.iloc[0])

    # for each label of 2_way_label calculate the mean, max and min length of the text in clean_title and write in a file
    with open(f'results/{dataset_name}/' + '2_way_label.txt', 'w') as f:
        for label in df["2_way_label"].unique():
            f.write(f"LABEL: {label}\n")
            f.write(f"MAX: {df[df['2_way_label'] == label]['clean_title'].str.len().max()}\n")
            f.write(f"MIN: {df[df['2_way_label'] == label]['clean_title'].str.len().min()}\n")
            f.write(f"MEAN: {df[df['2_way_label'] == label]['clean_title'].str.len().mean()}\n\n")
            # write also the number of rows with that label
            f.write(f"COUNT: {df[df['2_way_label'] == label].shape[0]}\n\n")
            
    # for each label of 3_way_label calculate the mean, max and min length of the text in clean_title and write in a file
    with open(f'results/{dataset_name}/' + '3_way_label.txt', 'w') as f:
        for label in df["3_way_label"].unique():
            f.write(f"LABEL: {label}\n")
            f.write(f"MAX: {df[df['3_way_label'] == label]['clean_title'].str.len().max()}\n")
            f.write(f"MIN: {df[df['3_way_label'] == label]['clean_title'].str.len().min()}\n")
            f.write(f"MEAN: {df[df['3_way_label'] == label]['clean_title'].str.len().mean()}\n\n")
            # write also the number of rows with that label
            f.write(f"COUNT: {df[df['3_way_label'] == label].shape[0]}\n\n")
    # for each label of 6_way_label calculate the mean, max and min length of the text in clean_title and write in a file
    with open(f'results/{dataset_name}/' + '6_way_label.txt', 'w') as f:
        for label in df["6_way_label"].unique():
            f.write(f"LABEL: {label}\n")
            f.write(f"MAX: {df[df['6_way_label'] == label]['clean_title'].str.len().max()}\n")
            f.write(f"MIN: {df[df['6_way_label'] == label]['clean_title'].str.len().min()}\n")
            f.write(f"MEAN: {df[df['6_way_label'] == label]['clean_title'].str.len().mean()}\n\n")
            # write also the number of rows with that label
            f.write(f"COUNT: {df[df['6_way_label'] == label].shape[0]}\n\n")


    for label in df["2_way_label"].unique():
        print(f"LABEL: {label}")
        print(f"COUNT: {df[df['2_way_label'] == label].shape[0]}\n\n")


    # print values of subreddit collum
    print(df["subreddit"].value_counts())
    for subreddit in df["subreddit"].unique():
        # print score of the subreddit
        print(f"SUBREDDIT: {subreddit}")
        print(f"MAX: {df[df['subreddit'] == subreddit]['score'].max()}")
        print(f"MIN: {df[df['subreddit'] == subreddit]['score'].min()}")
        print(f"MEAN: {df[df['subreddit'] == subreddit]['score'].mean()}")
        print("\n")

    # SAVE DATA
    if not os.path.exists(f'results/Fakeddit'):
        os.makedirs(f'results/Fakeddit')

    with open(f'results/Fakeddit/' + 'subreddit.txt', 'w') as f:
        f.write(f"{df['subreddit'].value_counts()}\n")
        for subreddit in df["subreddit"].unique():
            f.write(f"SUBREDDIT: {subreddit}\n")
            f.write(f"MAX: {df[df['subreddit'] == subreddit]['score'].max()}\n")
            f.write(f"MIN: {df[df['subreddit'] == subreddit]['score'].min()}\n")
            f.write(f"MEAN: {df[df['subreddit'] == subreddit]['score'].mean()}\n\n")
    
   

    # 2_way_label
    table = pd.crosstab(df["subreddit"], df["2_way_label"])
    plt.figure(figsize=(6, 10))
    sns.heatmap(table, cmap="Blues", fmt="d", linewidths=0.2, linecolor="gray", cbar=False)
    plt.title("Contingency Table 2_way_label")
    plt.xlabel("2_way_label")
    plt.ylabel("Subreddit")
    plt.tight_layout()
    table.to_csv(f'results/Fakeddit/' + 'subreddit_distribution_2_way.csv')
    plt_save_image_pdf(plt, f'results/Fakeddit/' + 'subreddit_distribution_2_way')

    # 3_way_label
    table = pd.crosstab(df["subreddit"], df["3_way_label"])
    plt.figure(figsize=(6, 10))
    sns.heatmap(table, cmap="Blues", fmt="d", linewidths=0.2, linecolor="gray", cbar=False)
    plt.title("Contingency Table 3_way_label")
    plt.xlabel("3_way_label")
    plt.ylabel("Subreddit")
    plt.tight_layout()
    table.to_csv(f'results/Fakeddit/' + 'subreddit_distribution_3_way.csv')
    plt_save_image_pdf(plt, f'results/Fakeddit/' + 'subreddit_distribution_3_way')

    # 6_way_label
    table = pd.crosstab(df["subreddit"], df["6_way_label"])
    plt.figure(figsize=(6, 10))
    sns.heatmap(table, cmap="Blues", fmt="d", linewidths=0.2, linecolor="gray", cbar=False)
    plt.title("Contingency Table 6_way_label")
    plt.xlabel("6_way_label")
    plt.ylabel("Subreddit")
    plt.tight_layout()
    table.to_csv(f'results/Fakeddit/' + 'subreddit_distribution_6_way.csv')
    plt_save_image_pdf(plt, f'results/Fakeddit/' + 'subreddit_distribution_6_way')

    fig, axes = plt.subplots(1, 3, figsize=(8, 6))  # 1 row, 3 columns
    plt.suptitle("Fakeddit", fontsize=14, fontweight="bold")

    # List of labels for iteration
    labels = ["2_way_label", "3_way_label", "6_way_label"]
    for i, label in enumerate(labels):
        table = pd.crosstab(df["subreddit"], df[label])
        if i == 0:
            sns.heatmap(table, cmap="Blues", fmt="d", linewidths=0.2, linecolor="gray", cbar=False, ax=axes[i])
        elif i == 1:
            sns.heatmap(table, cmap="Blues", fmt="d", linewidths=0.2, linecolor="gray", cbar=False, ax=axes[i])
            axes[i].set_yticklabels([])
            axes[i].tick_params(left=False)
        else:
            sns.heatmap(table, cmap="Blues", fmt="d", linewidths=0.2, linecolor="gray", cbar=True, ax=axes[i])
            axes[i].set_yticklabels([])
            axes[i].tick_params(left=False)

        # Set title and labels
        axes[i].set_xlabel(label)
        # Save table to CSV
        table.to_csv(f'results/Fakeddit/subreddit_distribution_{label}.csv')
    # Adjust layout and save figure
    rect = plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color="black", linewidth=2, fill=False)
    fig.patches.append(rect)
    plt.tight_layout()
    plt_save_image_pdf(plt, f'results/Fakeddit/' + 'subreddit_distribution_way')
    
    
def fauxtography():
    df = pd.read_csv("datasets/Fauxtography-fake-image-detection/data/processed/reuters/media_2019-03-23 20:06:30.csv")
    #df = pd.read_csv("datasets/Fauxtography-fake-image-detection/data/processed/snopes/media_2019-02-02 15:59:21.csv")
    
    dataset_name = "Fauxtography"
    print(f"{dataset_name} INFO")
    print(df.columns)
    # print first row 
    print(df.iloc[1])
    print(df["label"].value_counts())
    # PRINT SHAPE   
    print(df.shape)
    # GET the highest and lowest value in the collum "false_perc"
    print(df["false_perc"].max())
    print(df["false_perc"].min())
    # GET the highest and lowest value in the collum "true_perc"
    print(df["true_perc"].max())
    print(df["true_perc"].min())
    # GET the highest and lowest value in the collum "mixed_perc"
    print(df["mixed_perc"].max())
    print(df["mixed_perc"].min())
    
    # print the len of the min claim and max claim
    print(f"MIN: {df['claim'].str.len().min()}")
    print(f"MAX: {df['claim'].str.len().max()}")
    print(f"MEAN: {df['claim'].str.len().mean()}")   
    # SAVE DATA
    if not os.path.exists(f'results/{dataset_name}'):
        os.makedirs(f'results/{dataset_name}')
    #with open(f'results/{dataset_name}/' + 'TRUE.txt', 'w') as f:
    with open(f'results/{dataset_name}/' + 'FALSE.txt', 'w') as f:
        f.write(f"{df['label'].value_counts()}\n")
        for label in df["label"].unique(): 
            f.write(f"LABEL: {label}\n")
            f.write(f"MAX: {df[df['label'] == label]['claim'].str.len().max()}\n")
            f.write(f"MIN: {df[df['label'] == label]['claim'].str.len().min()}\n")
            f.write(f"MEAN: {df[df['label'] == label]['claim'].str.len().mean()}\n\n")


def CovID():
    df_CovID1 = pd.read_csv("datasets/CovID/CovIDI/D1.csv", encoding='latin1')
    df_CovID2 = pd.read_csv("datasets/CovID/CovIDII/D2.csv", encoding='latin1')

    print("CovIDI")
    print(f"Shape: {df_CovID1.shape}")
    print(f"Collumns: {df_CovID1.columns}")
    print("CovIDII")
    print(f"Shape: {df_CovID2.shape}")
    print(f"Collumns of CovIDII: {df_CovID2.columns}")

    # check if there are elements in Title column of df_CovID2 that are or in the Title column or in Title1 colum of df_CovID1
    # print("Checking for common elements in Title column of CovIDII and CovIDI")
    # common_elements = df_CovID2['Title'].isin(df_CovID1['Title'])
    # print(f"Common elements: {common_elements.sum()}")
    # # print common elements
    # # print(df_CovID2[common_elements])
    # print("Checking for common elements in Title1 column of CovIDII and CovIDI")
    # common_elements = df_CovID2['Title'].isin(df_CovID1['Title1'])
    # print(f"Common elements: {common_elements.sum()}")
    
    print("Checking for common elements in Text column of CovIDII and CovIDI")
    common_elements = df_CovID2['Text'].isin(df_CovID1['Text'])
    print(f"Common elements: {common_elements.sum()}")

    # SAVE DATA 
    
            
    for dataset_name, df in zip(['CovID1', 'CovID2'], [df_CovID1, df_CovID2]):
        if not os.path.exists(f'results/{dataset_name}'):
            os.makedirs(f'results/{dataset_name}')
            # crete a 0_NOTES.txt file with Text common elements
            with open(f'results/{dataset_name}/' + '0_NOTES.txt', 'w') as f:
                f.write(f"SHAPE: {df.shape}\n")
                f.write("Common elements in Text column of CovIDII and CovIDI\n")
                f.write(f"Common elements: {common_elements.sum()}\n")
                

        with open(f'results/{dataset_name}/' + 'label_distribution.txt', 'w') as f:
            f.write(f"LABELS\n{df['Label'].value_counts()}\n\n")
            for label in df["Label"].unique():
                f.write(f"LABEL: {label}\n")
                f.write(f"MAX: {df[df['Label'] == label]['Text'].str.len().max()}\n")
                f.write(f"MIN: {df[df['Label'] == label]['Text'].str.len().min()}\n")
                f.write(f"MEAN: {df[df['Label'] == label]['Text'].str.len().mean()}\n")
                f.write(f"% of the total: {round(df['Label'].value_counts()[label] / df.shape[0], 4)}\n\n")


def evons():
    dataset_name = 'evons'
    df = pd.read_csv("datasets/evons/evons.csv")
    print(f"Shape: {df.shape}")
    print(f"Collumns: {df.columns}")
    # check values of the is_valid_image column
    print(df['is_valid_image'].value_counts())
    # keep in df only the rows with is_valid_image = 1
    df = df[df['is_valid_image'] == 1]
    print(f"Shape after removing invalid images: {df.shape}")
    if not os.path.exists(f'results/{dataset_name}'):
        os.makedirs(f'results/{dataset_name}')
        with open(f'results/{dataset_name}/' + '0_NOTES.txt', 'w') as f:
            f.write(f"SHAPE MULTIMODAL WITH IMAGE: {df.shape}\n")
            f.write(f"LABELS\n{df['is_fake'].value_counts()}\n\n")
    
    with open(f'results/{dataset_name}/' + 'label_distribution.txt', 'w') as f:
        f.write(f"LABELS\n{df['is_fake'].value_counts()}\n\n")
        for label in df["is_fake"].unique():
            f.write(f"LABEL: {label}\n")
            f.write(f"MAX: {df[df['is_fake'] == label]['description'].str.len().max()}\n")
            f.write(f"MIN: {df[df['is_fake'] == label]['description'].str.len().min()}\n")
            f.write(f"MEAN: {df[df['is_fake'] == label]['description'].str.len().mean()}\n")
            f.write(f"% of the total: {round(df['is_fake'].value_counts()[label] / df.shape[0], 4)}\n\n")


def factify2():
    dataset_name = "factify2"
    print(f"{dataset_name} INFO")
    df_train = pd.read_csv('datasets/factify2/train/train.csv', index_col=0, sep='\t')[['claim', 'claim_image', 'document', 'document_image', 'Category', 'Claim OCR', 'Document OCR']]
    df_val = pd.read_csv('datasets/factify2/val/val.csv', index_col=0, sep='\t')[['claim', 'claim_image', 'document', 'document_image', 'Category', 'Claim OCR', 'Document OCR']]
    df_concact = pd.concat([df_train, df_val])
    # for SET in ['train', 'val', 'test']:

    #     if SET != 'test':
    #         df = pd.read_csv('datasets/factify2/' + SET + '/' + SET + '.csv', index_col=0, sep='\t')[['claim', 'claim_image', 'document', 'document_image', 'Category', 'Claim OCR', 'Document OCR']]
    #     else:
    #         df = pd.read_csv('datasets/factify2/' + SET + '/' + SET + '.csv', index_col=0, sep='\t')[['claim', 'claim_image', 'document', 'document_image', 'Claim OCR', 'Document OCR']]

    #     print(f"{SET} INFO")
        # print(df.columns)
    print(df_concact.shape)
    if not os.path.exists(f'results/{dataset_name}'):
        os.makedirs(f'results/{dataset_name}')
    with open(f'results/{dataset_name}/' + 'Claim_Category.txt', 'w') as f:
        for label in df_concact["Category"].unique():
            f.write(f"LABEL: {label}\n")
            f.write(f"COUNT: {df_concact[df_concact['Category'] == label].shape[0]}\n")
            # mean, max and min length of the text in claim
            f.write(f"MAX: {df_concact[df_concact['Category'] == label]['claim'].str.len().max()}\n")
            f.write(f"MIN: {df_concact[df_concact['Category'] == label]['claim'].str.len().min()}\n")
            f.write(f"MEAN: {df_concact[df_concact['Category'] == label]['claim'].str.len().mean()}\n\n")

    # same for document
    with open(f'results/{dataset_name}/' + 'Document_Categoty.txt', 'w') as f:
        for label in df_concact["Category"].unique():
            f.write(f"LABEL: {label}\n")
            f.write(f"COUNT: {df_concact[df_concact['Category'] == label].shape[0]}\n")
            # mean, max and min length of the text in claim
            f.write(f"MAX: {df_concact[df_concact['Category'] == label]['document'].str.len().max()}\n")
            f.write(f"MIN: {df_concact[df_concact['Category'] == label]['document'].str.len().min()}\n")
            f.write(f"MEAN: {df_concact[df_concact['Category'] == label]['document'].str.len().mean()}\n\n")

    # get the id row with the max length of document entry
    print(df_concact[df_concact['document'].str.len() == df_concact['document'].str.len().max()]['document'])
    with open(f'results/{dataset_name}/' + 'max_document.txt', 'w') as f:
        f.write(f"{df_concact[df_concact['document'].str.len() == df_concact['document'].str.len().max()]['document']}\n")


    desired_order = ["Support_Multimodal", "Support_Text", "Insufficient_Multimodal", "Insufficient_Text", "Refute"]
    categories = [cat for cat in desired_order if cat in df_concact['Category'].unique()]
    colors = ['green', 'yellow', 'orange', 'brown', 'red']
    claim_bins = np.linspace(0, 750, 50)  # 50 bin tra 0 e 800
    document_bins = np.linspace(0, 300000, 50)  # 50 bin tra 0 e 140000
    fig, axes = plt.subplots(2, len(categories), figsize=(15, 7))
    fig.tight_layout(pad=3.0)
    fig.subplots_adjust(hspace=0.5)
    for i, category in enumerate(categories):
        sub_df = df_concact[df_concact['Category'] == category]
        claim_lengths = sub_df['claim'].astype(str).apply(len)
        mean_claim_length = claim_lengths.mean()
        
        axes[0, i].hist(claim_lengths, bins=claim_bins, color=colors[i], alpha=0.7, edgecolor='black', log=True)
        axes[0, i].axvline(mean_claim_length, color='black', linestyle='dashed', linewidth=1)
        axes[0, i].set_title(f"{category}\nMean: {mean_claim_length:.2f}")
        axes[0, i].set_xlabel("Claim Length")
        axes[0, i].set_ylabel("Frequency")
        axes[0, i].set_xlim(0, 800)  
        axes[0, i].set_ylim(1, 10**4) 
        document_lengths = sub_df['document'].astype(str).apply(len)
        mean_document_length = document_lengths.mean()
        
        axes[1, i].hist(document_lengths, bins=document_bins, color=colors[i], alpha=0.7, edgecolor='black', log=True)
        axes[1, i].axvline(mean_document_length, color='black', linestyle='dashed', linewidth=1)
        axes[1, i].set_title(f"{category}\nMean: {mean_document_length:.2f}")
        axes[1, i].set_xlabel("Document Length")
        axes[1, i].set_ylabel("Frequency")
    plt_save_image_pdf(plt, f'results/{dataset_name}/Claim_Document_Length')


def ReCOVery():
    dataset_name = 'ReCOVery'
    df = pd.read_csv("datasets/ReCOVery/dataset/recovery-news-data.csv")
    print(f"Shape: {df.shape}")
    print(f"Collumns: {df.columns}")
    if not os.path.exists(f'results/{dataset_name}'):
        os.makedirs(f'results/{dataset_name}')
        with open(f'results/{dataset_name}/' + '0_NOTES.txt', 'w') as f:
            f.write(f"SHAPE: {df.shape}\n")

    with open(f'results/{dataset_name}/' + 'label_distribution.txt', 'w') as f:
        f.write(f"LABELS\n{df['reliability'].value_counts()}\n\n")
        for label in df["reliability"].unique():
            f.write(f"LABEL: {label}\n")
            f.write(f"MAX: {df[df['reliability'] == label]['body_text'].str.len().max()}\n")
            f.write(f"MIN: {df[df['reliability'] == label]['body_text'].str.len().min()}\n")
            f.write(f"MEAN: {df[df['reliability'] == label]['body_text'].str.len().mean()}\n")
            f.write(f"% of the total: {round(df['reliability'].value_counts()[label] / df.shape[0], 4)}\n\n")


if __name__ == "__main__":
    fakeddit()