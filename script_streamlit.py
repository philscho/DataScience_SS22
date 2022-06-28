"""
Data Science 1, Sommersemester 2022, Goethe Universität Frankfurt
Philipp Scholl (5385416)
Topic: PageRank on a disease association graph
"""

import networkx as nx
import streamlit as st
from numpy import linalg as LA
import pandas as pd

st.title('PageRank on disease association network')
st.write("by Philipp Scholl (philipp.scholl@stud.uni-frankfurt.de)")
st.write("Data source: http://snap.stanford.edu/biodata/datasets/10006/10006-DD-Miner.html" + 
         " " + "http://snap.stanford.edu/biodata/datasets/10021/10021-D-DoMiner.html")
left_column, right_column = st.columns(2)

directory_path = "C:/Users/philscho/Documents/Studium/SoSe 22/Data Science/Aufgabe/Datasets/"

graphdata_filepath = directory_path + "DD-Miner_miner-disease-disease.tsv"
descrdata_filepath = directory_path + "D-DoMiner_miner-diseaseDOID.tsv"

# Construct networkx graph object from file (edgelist)    
fh = open(graphdata_filepath)
G = nx.read_edgelist(fh, delimiter='\t')
fh.close()
# Pandas dataframe of edgelist for visualization
df_edges = nx.to_pandas_edgelist(G)

# Dataframe of disease entity descriptions
df_disease_descr = pd.read_table(descrdata_filepath)

right_column.subheader("Disease descriptions:")
right_column.dataframe(df_disease_descr)

# Attach disease names to edgelist dataframe
df_edges_ = pd.merge(df_edges, df_disease_descr, how='left', left_on='source', 
                     right_on='# Disease(DOID)')
df_edges_.drop(['# Disease(DOID)', 'Definition', 'Synonym'], axis=1, 
               inplace=True)
df_edges_ = pd.merge(df_edges_, df_disease_descr, how='left', left_on='target', 
                     right_on='# Disease(DOID)')
df_edges_.drop(['# Disease(DOID)', 'Definition', 'Synonym'], axis=1, 
               inplace=True)
df_edges_.rename(
    columns={"Name_x": "disease_source", "Name_y": "disease_target"},
    inplace=True)

left_column.subheader("Graph edgelist (with disease names)")
left_column.dataframe(df_edges_)

# Run standard PageRank algorithm on graph
pagerank = nx.pagerank(G)
# Dataframe of PageRank vector for visualization
df_pr = pd.DataFrame(list(pagerank.items()), columns=['node', 'pagerank'])

# Select nodes of specific disease type for personalization
specific_disease = "gastritis"
df_specific_diseases = df_disease_descr[
    df_disease_descr["Name"].str.contains(specific_disease)==True]

# Create personalization dictionary with uniformly distributed values
personalization_dict = {node: 1.0 / len(df_specific_diseases) 
                        for node in df_specific_diseases["# Disease(DOID)"]}

# Calculate personalized PageRank vector and get DataFrame
ppr = nx.pagerank(G, personalization=personalization_dict)
df_ppr = pd.DataFrame(list(ppr.items()), columns=['node', 'pagerank'])

left_column.subheader("Nodes of personalization vector:")
left_column.dataframe(df_specific_diseases)

# Merge PageRank dataframes
df_both_pr = pd.merge(df_pr, df_ppr, on='node')
df_both_pr.rename(
    columns={"pagerank_x": "PageRank", "pagerank_y": "Personalized PageRank"},
    inplace=True)
# Add column for difference between PageRank values
df_both_pr["Difference"] = abs(df_both_pr["PageRank"] - 
                               df_both_pr["Personalized PageRank"])

# Euclidean distance between PageRank vectors
l2_dist = LA.norm(df_both_pr['PageRank'] - 
                  df_both_pr['Personalized PageRank'], ord=2)

# Add disease description to PageRank dataframe
df_both_pr_ = pd.merge(df_both_pr, df_disease_descr, how='left', left_on='node', right_on='# Disease(DOID)')
df_both_pr_.drop(['# Disease(DOID)', 'Definition', 'Synonym'], axis=1, inplace=True)

right_column.subheader("PageRank:")
right_column.dataframe(df_both_pr_)

right_column.metric(label="Euclidean distance", value=l2_dist)
