import streamlit as st
import requests
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
import math
import datetime
import time
from collections import OrderedDict
import io

# Base URLs for PubMed E-utilities
BASEURL_SRCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
BASEURL_FTCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
DB = 'pubmed'
BATCH_NUM = 1000  # Number of records to fetch per batch

def mkquery(base_url, params):
    query = '&'.join([f'{key}={urllib.parse.quote(str(value))}' for key, value in params.items()])
    url = f"{base_url}?{query}"
    return url

def getXmlFromURL(base_url, params):
    url = mkquery(base_url, params)
    response = requests.get(url)
    return ET.fromstring(response.text)

def getTextFromNode(root, path, fill='', mode=0, attrib=''):
    node = root.find(path)
    if node is None:
        return fill
    else:
        return node.text if mode == 0 and node.text is not None else node.get(attrib, fill)

def pushData(rootXml, articles_list):
    for article in rootXml.findall('.//PubmedArticle'):
        data = OrderedDict()
        data['PMID'] = getTextFromNode(article, './MedlineCitation/PMID', '')
        data['JournalTitle'] = getTextFromNode(article, './MedlineCitation/Article/Journal/Title', '')
        data['ArticleTitle'] = getTextFromNode(article, './MedlineCitation/Article/ArticleTitle', '')
        
        abstract_elems = article.findall('./MedlineCitation/Article/Abstract/AbstractText')
        if abstract_elems:
            data['Abstract'] = ' '.join([elem.text.strip() for elem in abstract_elems if elem.text])
        else:
            data['Abstract'] = ''
        
        pub_year = getTextFromNode(article, './MedlineCitation/Article/ArticleDate/Year', '')
        if not pub_year:
            pub_year = getTextFromNode(article, './MedlineCitation/Article/Journal/JournalIssue/PubDate/Year', '')
        data['PublicationYear'] = pub_year
        
        articles_list.append(data)

def fetch_pubmed_articles(query, batch_size=BATCH_NUM, progress_callback=None):
    params = {
        'db': DB,
        'term': query,
        'usehistory': 'y',
        'retmax': 0
    }
    root = getXmlFromURL(BASEURL_SRCH, params)
    count = int(getTextFromNode(root, './Count', '0'))
    articles = []
    if count == 0:
        return articles
    
    query_key = getTextFromNode(root, './QueryKey', '')
    webenv = getTextFromNode(root, './WebEnv', '')
    iterCount = math.ceil(count / batch_size)
    
    for i in range(iterCount):
        params_fetch = {
            'db': DB,
            'query_key': query_key,
            'WebEnv': webenv,
            'retstart': i * batch_size,
            'retmax': batch_size,
            'retmode': 'xml'
        }
        root_fetch = getXmlFromURL(BASEURL_FTCH, params_fetch)
        pushData(root_fetch, articles)
        if progress_callback:
            progress_callback((i + 1) / iterCount)
        time.sleep(0.34)
    return articles

def to_excel(df):
    output = io.BytesIO()
    # Use context manager to automatically close the writer
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
         df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

st.title("PubMed Articles Fetcher and Excel Generator")
st.write("Enter the details below to fetch PubMed articles for periods before and after FDA approval, and then generate Excel files for download.")

with st.form("input_form"):
    drug_name = st.text_input("Drug Name", value="Quviviq")
    composition = st.text_input("Composition", value="daridorexant")
    disease = st.text_input("Disease", value="Insomnia")
    fda_approval_date = st.date_input("FDA Approval Date", value=datetime.date(2022, 1, 7))
    company = st.text_input("Company", value="Idorsia Pharmaceuticals US Inc")
    submitted = st.form_submit_button("Fetch Articles")

if submitted:
    with st.spinner("Fetching articles from PubMed..."):
        fda_date = datetime.datetime.combine(fda_approval_date, datetime.datetime.min.time())
        before_start_date = fda_date.replace(year=fda_date.year - 10)
        before_end_date = fda_date
        after_start_date = fda_date
        after_end_date = datetime.datetime.today()
        
        before_start_date_str = before_start_date.strftime("%Y/%m/%d")
        before_end_date_str = before_end_date.strftime("%Y/%m/%d")
        after_start_date_str = after_start_date.strftime("%Y/%m/%d")
        after_end_date_str = after_end_date.strftime("%Y/%m/%d")
        
        query_before = (f'"{composition}"[All Fields] AND "{disease}"[All Fields] '
                        f'AND ("{before_start_date_str}"[pdat] : "{before_end_date_str}"[pdat])')
        query_after = (f'("{drug_name}"[All Fields] OR "{composition}"[All Fields]) AND '
                       f'"{disease}"[All Fields] AND "{company}"[All Fields] '
                       f'AND ("{after_start_date_str}"[pdat] : "{after_end_date_str}"[pdat])')
        
        st.write("**Query for Before Approval Articles:**")
        st.code(query_before)
        st.write("**Query for After Approval Articles:**")
        st.code(query_after)
        
        st.write("Fetching articles for Before Approval period...")
        progress_bar_before = st.progress(0)
        articles_before = fetch_pubmed_articles(query_before, progress_callback=lambda progress: progress_bar_before.progress(int(progress * 100)))
        
        st.write("Fetching articles for After Approval period...")
        progress_bar_after = st.progress(0)
        articles_after = fetch_pubmed_articles(query_after, progress_callback=lambda progress: progress_bar_after.progress(int(progress * 100)))
    
    df_before = pd.DataFrame(articles_before)
    df_after = pd.DataFrame(articles_after)
    
    excel_before = to_excel(df_before)
    excel_after = to_excel(df_after)
    
    st.success("Excel files generated successfully!")
    
    st.download_button(
        label="Download Excel for Before Approval Articles",
        data=excel_before,
        file_name=f"{drug_name}_BeforeApproval.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.download_button(
        label="Download Excel for After Approval Articles",
        data=excel_after,
        file_name=f"{drug_name}_AfterApproval.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Rhenix Life Sciences</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>© 2025 Rhenix Life Sciences. All rights reserved.</p>", unsafe_allow_html=True)