import streamlit as st
import requests
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import math
import datetime
import time
import re
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import OrderedDict
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from io import BytesIO

# Make sure nltk resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Set Streamlit page configuration
st.set_page_config(
    page_title="PubMed Research Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.25rem;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .insight-box {
        background-color: #E3F2FD;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .text-centered {
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #BBDEFB;
        color: #616161;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F8F9FA;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #BBDEFB;
    }
</style>
<h1 class="main-header">PubMed Research Analyzer</h1>
<p class="text-centered">Extract, analyze, and download PubMed articles before and after FDA approval.</p>
""", unsafe_allow_html=True)

# ------------------ PubMed Retrieval Functions ------------------

DB = 'pubmed'
BASEURL_SRCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
BASEURL_FTCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
BATCH_NUM = 1000  # records per batch

def mkquery(base_url, params):
    """Build a URL with proper URL encoding."""
    query = '&'.join([f'{key}={urllib.parse.quote(str(value))}' for key, value in params.items()])
    return f"{base_url}?{query}"

def getXmlFromURL(base_url, params):
    """Return the XML ElementTree for a given URL built from parameters."""
    url = mkquery(base_url, params)
    response = requests.get(url)
    return ET.fromstring(response.text)

def getTextFromNode(root, path, default=""):
    """Safely extract text from an XML node specified by an XPath."""
    node = root.find(path)
    return node.text.strip() if node is not None and node.text is not None else default

# ------------------ Processing Each Article ------------------

def process_article(article):
    """
    Given an XML element for a PubMed article, extract basic information and author details
    """
    record = OrderedDict()
    record["PMID"] = getTextFromNode(article, "./MedlineCitation/PMID", "")
    record["JournalTitle"] = getTextFromNode(article, "./MedlineCitation/Article/Journal/Title", "")
    record["ArticleTitle"] = getTextFromNode(article, "./MedlineCitation/Article/ArticleTitle", "")
    
    pub_year = getTextFromNode(article, "./MedlineCitation/Article/ArticleDate/Year", "")
    if not pub_year:
        pub_year = getTextFromNode(article, "./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year", "")
    record["PublicationYear"] = pub_year
    
    abstracts = article.findall("./MedlineCitation/Article/Abstract/AbstractText")
    if abstracts:
        record["Abstract"] = " ".join([a.text.strip() for a in abstracts if a.text])
    else:
        record["Abstract"] = ""
    
    authors = article.findall("./MedlineCitation/Article/AuthorList/Author")
    author_count = 1
    
    for auth in authors:
        name = ""
        collective = auth.find("CollectiveName")
        if collective is not None and collective.text:
            name = collective.text.strip()
        else:
            fore = auth.find("ForeName")
            last = auth.find("LastName")
            if fore is not None and last is not None:
                name = f"{fore.text.strip()} {last.text.strip()}"
            elif fore is not None:
                name = fore.text.strip()
            elif last is not None:
                name = last.text.strip()
        
        aff_elem = auth.find("./AffiliationInfo/Affiliation")
        affiliation = aff_elem.text.strip() if (aff_elem is not None and aff_elem.text) else ""
        
        record[f"Author{author_count}"] = name
        record[f"Affiliation{author_count}"] = affiliation
        author_count += 1
    
    return record

def fetch_pubmed_articles(query, batch_size=BATCH_NUM, limit=None, progress_callback=None):
    """
    Use ESearch and EFetch to retrieve PubMed articles matching a query.
    Returns a list of OrderedDict records (one per article), each processed by process_article().
    Optional limit parameter to restrict the number of articles processed.
    """
    params = {"db": DB, "term": query, "usehistory": "y", "retmax": 0}
    root = getXmlFromURL(BASEURL_SRCH, params)
    count = int(getTextFromNode(root, "./Count", "0"))
    
    if count == 0:
        return []
    
    # If limit is specified, adjust count
    if limit and limit < count:
        count = limit
    
    query_key = getTextFromNode(root, "./QueryKey", "")
    webenv = getTextFromNode(root, "./WebEnv", "")
    
    records = []
    iterCount = math.ceil(count / batch_size)
    for i in range(iterCount):
        params_fetch = {
            "db": DB,
            "query_key": query_key,
            "WebEnv": webenv,
            "retstart": i * batch_size,
            "retmax": batch_size,
            "retmode": "xml"
        }
        
        # For the last batch with a limit, adjust retmax
        if limit and (i+1) * batch_size > limit:
            params_fetch["retmax"] = limit - i * batch_size
        
        root_fetch = getXmlFromURL(BASEURL_FTCH, params_fetch)
        for art in root_fetch.findall(".//PubmedArticle"):
            record = process_article(art)
            records.append(record)
            
            # Check if we've reached the limit
            if limit and len(records) >= limit:
                break
                
        # If we've reached the limit, break out of the loop
        if limit and len(records) >= limit:
            break
            
        # Update progress
        if progress_callback:
            progress_callback((i + 1) / iterCount)
            
        time.sleep(0.34)  # Respect rate limits
    return records

# ------------------ Data Verification Functions ------------------

def check_date_range(df, dataset_type, start_date, end_date):
    """
    Check if dates are within the expected range.
    For records with missing or invalid dates, we'll keep them.
    Returns the count of invalid dates found.
    """
    invalid_count = 0
    rows_to_drop = []
    
    for idx, row in df.iterrows():
        # Skip if no publication year
        if pd.isna(row.get('PublicationYear', None)) or row.get('PublicationYear', '') == '':
            continue
            
        try:
            # Try to convert the year to integer
            year = int(row['PublicationYear'])
            
            # Create a date object (use January 1 as default)
            pub_date = datetime.datetime(year, 1, 1)
            
            # Check if date is outside the expected range
            if dataset_type == "before" and (pub_date < start_date or pub_date > end_date):
                rows_to_drop.append(idx)
                invalid_count += 1
            elif dataset_type == "after" and (pub_date < start_date):
                rows_to_drop.append(idx)
                invalid_count += 1
        except (ValueError, TypeError):
            # If year can't be converted to integer, keep the row
            continue
    
    # Drop rows with invalid dates
    if rows_to_drop:
        df.drop(rows_to_drop, inplace=True)
    
    return invalid_count

def find_max_author_column(df):
    """Find the maximum author column that has non-empty values"""
    max_author = 0
    author_pattern = re.compile(r"^Author(\d+)$")
    
    # For each row, find the highest author number with data
    for _, row in df.iterrows():
        row_max = 0
        for col in df.columns:
            match = author_pattern.match(col)
            if match:
                author_num = int(match.group(1))
                # Check if this specific cell has content
                if pd.notna(row[col]) and row[col] != '':
                    if author_num > row_max:
                        row_max = author_num
        
        # Update the overall max if this row has a higher number
        if row_max > max_author:
            max_author = row_max
    
    return max_author

def trim_author_columns(df, max_author):
    """Trim excess author columns beyond the maximum needed"""
    columns_to_keep = []
    author_pattern = re.compile(r"^Author(\d+)$")
    affiliation_pattern = re.compile(r"^Affiliation(\d+)$")
    profile_pattern = re.compile(r"^Profile(\d+)$")
    
    # Add all non-author/affiliation columns
    for col in df.columns:
        if (not author_pattern.match(col) and 
            not affiliation_pattern.match(col) and 
            not profile_pattern.match(col)):
            columns_to_keep.append(col)
    
    # Add author/affiliation columns up to max_author
    for i in range(1, max_author + 1):
        author_col = f"Author{i}"
        if author_col in df.columns:
            columns_to_keep.append(author_col)
        
        affiliation_col = f"Affiliation{i}"
        if affiliation_col in df.columns:
            columns_to_keep.append(affiliation_col)
        
        profile_col = f"Profile{i}"
        if profile_col in df.columns:
            columns_to_keep.append(profile_col)
    
    # Return the dataframe with only the needed columns
    return df[columns_to_keep]

def process_dataset(df, dataset_type, start_date, end_date):
    """Process a dataset to check dates and trim author columns"""
    original_row_count = len(df)
    original_column_count = len(df.columns)
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1. Check and filter by date if within range
    invalid_date_count = check_date_range(df, dataset_type, start_date, end_date)
    
    # 2. Trim excess author columns
    max_author_column = find_max_author_column(df)
    df = trim_author_columns(df, max_author_column)
    
    return df, {
        "original_rows": original_row_count,
        "original_columns": original_column_count,
        "final_rows": len(df),
        "final_columns": len(df.columns),
        "invalid_dates": invalid_date_count,
        "max_author": max_author_column,
        "columns_removed": original_column_count - len(df.columns)
    }

# ------------------ Data Analysis Functions ------------------

def create_top_journals_chart(df):
    """Create a bar chart of top journals by publication count"""
    if len(df) == 0 or 'JournalTitle' not in df.columns:
        return None
    
    journal_counts = df['JournalTitle'].value_counts().reset_index()
    journal_counts.columns = ['Journal', 'Count']
    
    # Get top 10 journals or all if less than 10
    top_n = min(10, len(journal_counts))
    top_journals = journal_counts.head(top_n)
    
    # Create horizontal bar chart
    fig = px.bar(
        top_journals, 
        y='Journal', 
        x='Count',
        orientation='h',
        title=f'Top {top_n} Journals by Publication Count',
        height=400,
        color='Count',
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    fig.update_layout(
        xaxis_title='Number of Publications',
        yaxis_title='Journal',
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig

def create_yearly_trend_chart(df):
    """Create a line chart showing publication trends over years"""
    if len(df) == 0 or 'PublicationYear' not in df.columns:
        return None
    
    # Convert publication year to numeric and create year count
    df_year = df.copy()
    df_year['PublicationYear'] = pd.to_numeric(df_year['PublicationYear'], errors='coerce')
    df_year = df_year.dropna(subset=['PublicationYear'])
    
    if len(df_year) == 0:
        return None
    
    # Round to integer years
    df_year['PublicationYear'] = df_year['PublicationYear'].astype(int)
    
    # Count by year
    year_counts = df_year['PublicationYear'].value_counts().reset_index()
    year_counts.columns = ['Year', 'Count']
    year_counts = year_counts.sort_values('Year')
    
    # Create line chart
    fig = px.line(
        year_counts, 
        x='Year', 
        y='Count',
        markers=True,
        title='Publication Trend by Year',
        height=400
    )
    
    fig.update_layout(
        xaxis_title='Publication Year',
        yaxis_title='Number of Publications'
    )
    
    return fig

def create_author_collaboration_network(df, top_n=20):
    """Create a simple author collaboration network visualization"""
    if len(df) == 0:
        return None
    
    # Get all author columns
    author_cols = [col for col in df.columns if col.startswith('Author') and col != 'Author']
    if not author_cols:
        return None
    
    # Collect all authors
    all_authors = []
    for _, row in df.iterrows():
        paper_authors = [row[col] for col in author_cols if pd.notna(row[col]) and row[col] != '']
        if len(paper_authors) > 1:  # Only consider papers with multiple authors
            all_authors.extend(paper_authors)
    
    # Count author occurrences
    author_counts = pd.Series(all_authors).value_counts()
    
    # Get top authors
    top_authors = author_counts.head(top_n).index.tolist()
    
    # Create edges between co-authors
    edges = []
    for _, row in df.iterrows():
        paper_authors = [row[col] for col in author_cols if pd.notna(row[col]) and row[col] != '' and row[col] in top_authors]
        for i in range(len(paper_authors)):
            for j in range(i+1, len(paper_authors)):
                edges.append((paper_authors[i], paper_authors[j]))
    
    # Count edge occurrences
    edge_counts = pd.Series(edges).value_counts().reset_index()
    if len(edge_counts) == 0:
        return None
        
    edge_counts.columns = ['pairs', 'weight']
    edge_counts[['source', 'target']] = pd.DataFrame(edge_counts['pairs'].tolist(), index=edge_counts.index)
    
    # Create nodes dataframe with author frequencies
    nodes = pd.DataFrame({
        'name': top_authors,
        'size': [author_counts[author] for author in top_authors]
    })
    
    # Create network visualization using Plotly
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    # Simple circular layout for nodes
    for i, (_, row) in enumerate(nodes.iterrows()):
        angle = 2 * math.pi * i / len(nodes)
        x = math.cos(angle)
        y = math.sin(angle)
        node_x.append(x)
        node_y.append(y)
        node_text.append(row['name'])
        node_size.append(row['size'] * 10)  # Scale node size
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_width = []
    
    for _, row in edge_counts.iterrows():
        source_idx = nodes[nodes['name'] == row['source']].index[0]
        target_idx = nodes[nodes['name'] == row['target']].index[0]
        
        x0, y0 = node_x[source_idx], node_y[source_idx]
        x1, y1 = node_x[target_idx], node_y[target_idx]
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_width.append(row['weight'])
    
    # Create plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=node_size,
            color='#1565C0',
            line=dict(width=1, color='#333')
        ),
        text=node_text,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title='Top Author Collaboration Network',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

def generate_wordcloud(df, column='Abstract'):
    """Generate a word cloud from text in a specified column"""
    if len(df) == 0 or column not in df.columns:
        return None
    
    # Combine all text
    text = ' '.join(df[column].dropna().astype(str))
    if not text or len(text) < 10:
        return None
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'disease', 'patient', 'treatment', 'study', 'use', 'result', 'method', 'conclusion', 'background', 'objective'}
    stop_words.update(custom_stopwords)
    
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        max_words=100,
        contour_width=1,
        contour_color='steelblue',
        colormap='Blues'
    ).generate(' '.join(filtered_words))
    
    # Convert to figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def generate_bigram_analysis(df, column='Abstract', top_n=15):
    """Generate bigram analysis from text"""
    if len(df) == 0 or column not in df.columns:
        return None
    
    # Combine all text
    texts = df[column].dropna().astype(str).tolist()
    if not texts:
        return None
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'disease', 'patient', 'treatment', 'study', 'use', 'result', 'data'}
    stop_words.update(custom_stopwords)
    
    # Initialize vectorizer for bigrams
    vectorizer = CountVectorizer(
        ngram_range=(2, 2),  # bigrams
        stop_words=stop_words,
        min_df=2  # minimum document frequency
    )
    
    # Get bigram counts
    X = vectorizer.fit_transform(texts)
    bigram_counts = np.asarray(X.sum(axis=0)).flatten()
    
    # Get bigram names and counts
    bigrams = [(word, count) for word, count in zip(vectorizer.get_feature_names_out(), bigram_counts)]
    bigrams.sort(key=lambda x: x[1], reverse=True)
    
    # Extract top N bigrams
    top_bigrams = bigrams[:top_n]
    
    # Create bar chart
    if not top_bigrams:
        return None
        
    bigram_df = pd.DataFrame(top_bigrams, columns=['Bigram', 'Count'])
    
    fig = px.bar(
        bigram_df, 
        y='Bigram', 
        x='Count', 
        orientation='h',
        title=f'Top {top_n} Bigrams in {column}',
        height=500,
        color='Count',
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    fig.update_layout(
        xaxis_title='Frequency',
        yaxis_title='Bigram',
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig

def compare_abstract_topics(df_before, df_after):
    """Compare abstract topics between before and after periods"""
    if len(df_before) == 0 or len(df_after) == 0 or 'Abstract' not in df_before.columns or 'Abstract' not in df_after.columns:
        return None
    
    # Get abstracts
    texts_before = df_before['Abstract'].dropna().astype(str).tolist()
    texts_after = df_after['Abstract'].dropna().astype(str).tolist()
    
    if not texts_before or not texts_after:
        return None
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'disease', 'patient', 'treatment', 'study', 'clinical', 'use', 'result', 'data'}
    stop_words.update(custom_stopwords)
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=1000,
        min_df=2
    )
    
    # Get combined TF-IDF
    all_texts = texts_before + texts_after
    vectorizer.fit(all_texts)
    
    # Transform each corpus
    X_before = vectorizer.transform(texts_before)
    X_after = vectorizer.transform(texts_after)
    
    # Get average TF-IDF for each term in each corpus
    avg_tfidf_before = np.asarray(X_before.mean(axis=0)).flatten()
    avg_tfidf_after = np.asarray(X_after.mean(axis=0)).flatten()
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Find differential terms (terms with largest difference in TF-IDF)
    diff_scores = []
    for i, term in enumerate(feature_names):
        diff = avg_tfidf_after[i] - avg_tfidf_before[i]
        diff_scores.append((term, diff))
    
    # Sort by absolute difference
    diff_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Get top differential terms
    top_n = 20
    top_diff = diff_scores[:top_n]
    
    # Create dataframe for visualization
    diff_df = pd.DataFrame(top_diff, columns=['Term', 'Difference'])
    diff_df['Period'] = ['After Approval' if d > 0 else 'Before Approval' for d in diff_df['Difference']]
    diff_df['Abs_Difference'] = diff_df['Difference'].abs()
    
    # Create horizontal bar chart
    fig = px.bar(
        diff_df, 
        y='Term', 
        x='Difference',
        orientation='h',
        color='Period',
        title=f'Top {top_n} Differential Terms Between Before and After Approval',
        height=600,
        color_discrete_map={
            'Before Approval': '#E53935',
            'After Approval': '#43A047'
        }
    )
    
    fig.update_layout(
        xaxis_title='TF-IDF Difference (After - Before)',
        yaxis_title='Term',
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig

def journal_distribution_pie(df):
    """Create a pie chart of journal distribution"""
    if len(df) == 0 or 'JournalTitle' not in df.columns:
        return None
    
    # Get journal counts
    journal_counts = df['JournalTitle'].value_counts().reset_index()
    journal_counts.columns = ['Journal', 'Count']
    
    # Keep top 8 journals, group others
    top_n = min(8, len(journal_counts))
    if len(journal_counts) > top_n:
        top_journals = journal_counts.head(top_n)
        other_count = journal_counts.iloc[top_n:]['Count'].sum()
        top_journals = pd.concat([
            top_journals, 
            pd.DataFrame({'Journal': ['Other Journals'], 'Count': [other_count]})
        ])
    else:
        top_journals = journal_counts
    
    # Create pie chart
    fig = px.pie(
        top_journals, 
        values='Count', 
        names='Journal',
        title='Distribution of Publications by Journal',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    
    return fig

def extract_statistical_insights(df_before, df_after):
    """Extract key statistical insights from the datasets"""
    insights = []
    
    # 1. Publication volume
    before_count = len(df_before)
    after_count = len(df_after)
    
    if before_count > 0 and after_count > 0:
        percent_change = ((after_count - before_count) / before_count) * 100
        change_direction = "increase" if percent_change > 0 else "decrease"
        
        insights.append(f"Publication volume showed a {abs(percent_change):.1f}% {change_direction} after FDA approval ({after_count} vs. {before_count} publications).")
    
    # 2. Journal diversity
    if 'JournalTitle' in df_before.columns and 'JournalTitle' in df_after.columns:
        unique_journals_before = df_before['JournalTitle'].nunique()
        unique_journals_after = df_after['JournalTitle'].nunique()
        
        if unique_journals_before > 0 and unique_journals_after > 0:
            journal_percent_change = ((unique_journals_after - unique_journals_before) / unique_journals_before) * 100
            journal_change_direction = "more diverse" if journal_percent_change > 0 else "less diverse"
            
            insights.append(f"Research became {journal_change_direction} after approval with {unique_journals_after} journals compared to {unique_journals_before} before.")
    
    # 3. Author count
    author_cols_before = [col for col in df_before.columns if col.startswith('Author')]
    author_cols_after = [col for col in df_after.columns if col.startswith('Author')]
    
    avg_authors_before = 0
    avg_authors_after = 0
    
    if author_cols_before:
        # Count non-empty author cells per row
        authors_per_paper_before = df_before[author_cols_before].notna().sum(axis=1)
        avg_authors_before = authors_per_paper_before.mean()
    
    if author_cols_after:
        authors_per_paper_after = df_after[author_cols_after].notna().sum(axis=1)
        avg_authors_after = authors_per_paper_after.mean()
    
    if avg_authors_before > 0 and avg_authors_after > 0:
        author_change = avg_authors_after - avg_authors_before
        author_direction = "more" if author_change > 0 else "fewer"
        
        insights.append(f"Research papers had {author_direction} authors on average after approval ({avg_authors_after:.1f} vs. {avg_authors_before:.1f} authors per paper).")
    
    # 4. Abstract length and complexity
    if 'Abstract' in df_before.columns and 'Abstract' in df_after.columns:
        # Average abstract length
        abstract_length_before = df_before['Abstract'].str.len().mean()
        abstract_length_after = df_after['Abstract'].str.len().mean()
        
        if not pd.isna(abstract_length_before) and not pd.isna(abstract_length_after):
            length_change = ((abstract_length_after - abstract_length_before) / abstract_length_before) * 100
            length_direction = "longer" if length_change > 0 else "shorter"
            
            insights.append(f"Abstracts became {abs(length_change):.1f}% {length_direction} after approval ({abstract_length_after:.0f} vs. {abstract_length_before:.0f} characters on average).")
    
    # 5. Publication year span
    if 'PublicationYear' in df_before.columns and 'PublicationYear' in df_after.columns:
        df_before_year = df_before.copy()
        df_after_year = df_after.copy()
        
        # Convert to numeric
        df_before_year['PublicationYear'] = pd.to_numeric(df_before_year['PublicationYear'], errors='coerce')
        df_after_year['PublicationYear'] = pd.to_numeric(df_after_year['PublicationYear'], errors='coerce')
        
        # Get year ranges
        before_min_year = df_before_year['PublicationYear'].min()
        before_max_year = df_before_year['PublicationYear'].max()
        after_min_year = df_after_year['PublicationYear'].min()
        after_max_year = df_after_year['PublicationYear'].max()
        
        if not pd.isna(before_min_year) and not pd.isna(before_max_year) and not pd.isna(after_min_year) and not pd.isna(after_max_year):
            insights.append(f"Before approval research spans {before_max_year - before_min_year + 1} years ({before_min_year:.0f}-{before_max_year:.0f}), while after approval spans {after_max_year - after_min_year + 1} years ({after_min_year:.0f}-{after_max_year:.0f}).")
    
    return insights

# ------------------ File Conversion Functions ------------------

def to_excel(df):
    """Convert dataframe to Excel file in memory"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        # Auto-adjust columns' width
        worksheet = writer.sheets['Sheet1']
        for i, col in enumerate(df.columns):
            column_width = max(df[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, column_width +.5)
        
    processed_data = output.getvalue()
    return processed_data

# ------------------ Main Streamlit App ------------------

# Initialize session state
if 'df_before' not in st.session_state:
    st.session_state.df_before = None
if 'df_after' not in st.session_state:
    st.session_state.df_after = None
if 'stats_before' not in st.session_state:
    st.session_state.stats_before = None
if 'stats_after' not in st.session_state:
    st.session_state.stats_after = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'excel_before' not in st.session_state:
    st.session_state.excel_before = None
if 'excel_after' not in st.session_state:
    st.session_state.excel_after = None

# Sidebar configuration
st.sidebar.markdown("### Search Configuration")

with st.sidebar.form("search_form"):
    drug_name = st.text_input("Drug Name", value="Quviviq")
    composition = st.text_input("Composition/Compound", value="daridorexant")
    disease = st.text_input("Target Disease", value="Insomnia")
    company = st.text_input("Company", value="Idorsia Pharmaceuticals US Inc")
    fda_approval_date = st.date_input("FDA Approval Date", value=datetime.date(2022, 1, 7))
    include_company = st.checkbox("Include company in search query", value=False)
    article_limit = st.number_input("Limit articles (0 for no limit)", min_value=0, value=0)
    
    submitted = st.form_submit_button("Search PubMed")

# Process when form is submitted
if submitted:
    with st.spinner("Fetching articles from PubMed..."):
        # Convert dates
        fda_date = datetime.datetime.combine(fda_approval_date, datetime.datetime.min.time())
        before_start_date = fda_date.replace(year=fda_date.year - 10)
        before_end_date = fda_date
        after_start_date = fda_date
        after_end_date = datetime.datetime.today()
        
        # Format dates for PubMed query
        before_start_date_str = before_start_date.strftime("%Y/%m/%d")
        before_end_date_str = before_end_date.strftime("%Y/%m/%d")
        after_start_date_str = after_start_date.strftime("%Y/%m/%d")
        after_end_date_str = after_end_date.strftime("%Y/%m/%d")
        
        # Build queries
        query_before = (f'"{composition}"[All Fields] AND "{disease}"[All Fields] '
                       f'AND ("{before_start_date_str}"[pdat] : "{before_end_date_str}"[pdat])')
        
        query_after = (f'("{drug_name}"[All Fields] OR "{composition}"[All Fields]) '
                      f'AND "{disease}"[All Fields] ')
        
        # Add company to query if requested
        if include_company:
            query_after += f'AND "{company}"[All Fields] '
            
        query_after += f'AND ("{after_start_date_str}"[pdat] : "{after_end_date_str}"[pdat])'
        
        st.subheader("Search Queries")
        st.code(f"Before Approval: {query_before}")
        st.code(f"After Approval: {query_after}")
        
        # Fetch articles
        st.write("Fetching articles for Before Approval period...")
        progress_bar_before = st.progress(0)
        limit_value = int(article_limit) if article_limit > 0 else None
        records_before = fetch_pubmed_articles(
            query_before, 
            limit=limit_value,
            progress_callback=lambda progress: progress_bar_before.progress(progress)
        )
        
        st.write("Fetching articles for After Approval period...")
        progress_bar_after = st.progress(0)
        records_after = fetch_pubmed_articles(
            query_after, 
            limit=limit_value,
            progress_callback=lambda progress: progress_bar_after.progress(progress)
        )
        
        # Convert to dataframes
        df_before_raw = pd.DataFrame(records_before)
        df_after_raw = pd.DataFrame(records_after)
        
        # Process and verify data
        st.subheader("Processing and Validating Data")
        
        if len(df_before_raw) > 0:
            st.write(f"Processing {len(df_before_raw)} articles from Before Approval period...")
            df_before, stats_before = process_dataset(df_before_raw, "before", before_start_date, before_end_date)
            st.session_state.df_before = df_before
            st.session_state.stats_before = stats_before
            st.session_state.excel_before = to_excel(df_before)
        else:
            st.warning("No articles found for Before Approval period.")
            st.session_state.df_before = pd.DataFrame()
            st.session_state.stats_before = {"original_rows": 0}
            st.session_state.excel_before = to_excel(pd.DataFrame())
        
        if len(df_after_raw) > 0:
            st.write(f"Processing {len(df_after_raw)} articles from After Approval period...")
            df_after, stats_after = process_dataset(df_after_raw, "after", after_start_date, after_end_date)
            st.session_state.df_after = df_after
            st.session_state.stats_after = stats_after
            st.session_state.excel_after = to_excel(df_after)
        else:
            st.warning("No articles found for After Approval period.")
            st.session_state.df_after = pd.DataFrame()
            st.session_state.stats_after = {"original_rows": 0}
            st.session_state.excel_after = to_excel(pd.DataFrame())
        
        # Generate insights
        st.session_state.insights = extract_statistical_insights(
            st.session_state.df_before, 
            st.session_state.df_after
        )
        
        st.success("Processing complete! You can now explore the analysis tabs below.")

# Display tabs for results
if st.session_state.df_before is not None or st.session_state.df_after is not None:
    tabs = st.tabs([
        "📊 Data Overview", 
        "📈 Time Trends", 
        "🔎 Content Analysis", 
        "👥 Author Analysis",
        "💾 Download Data"
    ])
    
    # Tab 1: Data Overview
    with tabs[0]:
        st.markdown("<h2 class='subheader'>Data Overview</h2>", unsafe_allow_html=True)
        
        # Display processing statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 class='section-header'>Before Approval</h3>", unsafe_allow_html=True)
            if st.session_state.stats_before:
                stats = st.session_state.stats_before
                st.write(f"Articles found: {stats.get('original_rows', 0)}")
                if stats.get('original_rows', 0) > 0:
                    st.write(f"Articles after date filtering: {stats.get('final_rows', 0)}")
                    st.write(f"Invalid dates removed: {stats.get('invalid_dates', 0)}")
                    st.write(f"Maximum author number: {stats.get('max_author', 0)}")
                    st.write(f"Columns trimmed: {stats.get('columns_removed', 0)}")
                
                # Preview the data
                if not st.session_state.df_before.empty:
                    st.write("Data Preview:")
                    st.dataframe(st.session_state.df_before.head(5), use_container_width=True)
        
        with col2:
            st.markdown("<h3 class='section-header'>After Approval</h3>", unsafe_allow_html=True)
            if st.session_state.stats_after:
                stats = st.session_state.stats_after
                st.write(f"Articles found: {stats.get('original_rows', 0)}")
                if stats.get('original_rows', 0) > 0:
                    st.write(f"Articles after date filtering: {stats.get('final_rows', 0)}")
                    st.write(f"Invalid dates removed: {stats.get('invalid_dates', 0)}")
                    st.write(f"Maximum author number: {stats.get('max_author', 0)}")
                    st.write(f"Columns trimmed: {stats.get('columns_removed', 0)}")
                
                # Preview the data
                if not st.session_state.df_after.empty:
                    st.write("Data Preview:")
                    st.dataframe(st.session_state.df_after.head(5), use_container_width=True)
        
        # Display key insights
        st.markdown("<h3 class='section-header'>Key Insights</h3>", unsafe_allow_html=True)
        
        if st.session_state.insights and len(st.session_state.insights) > 0:
            for insight in st.session_state.insights:
                st.markdown(f"<div class='insight-box'>💡 {insight}</div>", unsafe_allow_html=True)
        else:
            st.info("No significant insights could be generated from the data. Try adjusting your search parameters or including more articles.")
        
        # Journal distribution visualizations
        st.markdown("<h3 class='section-header'>Journal Distribution</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Before Approval:")
            if not st.session_state.df_before.empty:
                chart = create_top_journals_chart(st.session_state.df_before)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("Insufficient journal data for visualization.")
            else:
                st.info("No data available for visualization.")
        
        with col2:
            st.write("After Approval:")
            if not st.session_state.df_after.empty:
                chart = create_top_journals_chart(st.session_state.df_after)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("Insufficient journal data for visualization.")
            else:
                st.info("No data available for visualization.")
    
    # Tab 2: Time Trends
    with tabs[1]:
        st.markdown("<h2 class='subheader'>Publication Trends Over Time</h2>", unsafe_allow_html=True)
        
        # Publication trends over time
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 class='section-header'>Before Approval Trend</h3>", unsafe_allow_html=True)
            if not st.session_state.df_before.empty:
                chart = create_yearly_trend_chart(st.session_state.df_before)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("Insufficient year data for trend visualization.")
            else:
                st.info("No data available for trend visualization.")
        
        with col2:
            st.markdown("<h3 class='section-header'>After Approval Trend</h3>", unsafe_allow_html=True)
            if not st.session_state.df_after.empty:
                chart = create_yearly_trend_chart(st.session_state.df_after)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("Insufficient year data for trend visualization.")
            else:
                st.info("No data available for trend visualization.")
        
        # Journal distribution pie charts
        st.markdown("<h3 class='section-header'>Journal Distribution Over Time</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Before Approval:")
            if not st.session_state.df_before.empty:
                chart = journal_distribution_pie(st.session_state.df_before)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("Insufficient journal data for visualization.")
            else:
                st.info("No data available for visualization.")
        
        with col2:
            st.write("After Approval:")
            if not st.session_state.df_after.empty:
                chart = journal_distribution_pie(st.session_state.df_after)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("Insufficient journal data for visualization.")
            else:
                st.info("No data available for visualization.")
    
    # Tab 3: Content Analysis
    with tabs[2]:
        st.markdown("<h2 class='subheader'>Content Analysis</h2>", unsafe_allow_html=True)
        
        # Word clouds
        st.markdown("<h3 class='section-header'>Abstract Word Clouds</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Before Approval:")
            if not st.session_state.df_before.empty:
                fig = generate_wordcloud(st.session_state.df_before, column='Abstract')
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("Insufficient abstract data for word cloud.")
            else:
                st.info("No data available for word cloud.")
        
        with col2:
            st.write("After Approval:")
            if not st.session_state.df_after.empty:
                fig = generate_wordcloud(st.session_state.df_after, column='Abstract')
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("Insufficient abstract data for word cloud.")
            else:
                st.info("No data available for word cloud.")
        
        # Bigram analysis
        st.markdown("<h3 class='section-header'>Bigram Analysis</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Before Approval:")
            if not st.session_state.df_before.empty:
                chart = generate_bigram_analysis(st.session_state.df_before, column='Abstract')
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("Insufficient abstract data for bigram analysis.")
            else:
                st.info("No data available for bigram analysis.")
        
        with col2:
            st.write("After Approval:")
            if not st.session_state.df_after.empty:
                chart = generate_bigram_analysis(st.session_state.df_after, column='Abstract')
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("Insufficient abstract data for bigram analysis.")
            else:
                st.info("No data available for bigram analysis.")
        
        # Topic comparison
        st.markdown("<h3 class='section-header'>Topic Comparison</h3>", unsafe_allow_html=True)
        
        if not st.session_state.df_before.empty and not st.session_state.df_after.empty:
            chart = compare_abstract_topics(st.session_state.df_before, st.session_state.df_after)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("Insufficient data for topic comparison.")
        else:
            st.info("Data from both before and after approval periods is required for topic comparison.")
    
    # Tab 4: Author Analysis
    with tabs[3]:
        st.markdown("<h2 class='subheader'>Author Analysis</h2>", unsafe_allow_html=True)
        
        # Author network
        st.markdown("<h3 class='section-header'>Author Collaboration Networks</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Before Approval:")
            if not st.session_state.df_before.empty:
                top_n = st.slider("Number of top authors (Before)", min_value=5, max_value=50, value=15, key="before_author_slider")
                chart = create_author_collaboration_network(st.session_state.df_before, top_n=top_n)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("Insufficient author data for collaboration network.")
            else:
                st.info("No data available for author analysis.")
        
        with col2:
            st.write("After Approval:")
            if not st.session_state.df_after.empty:
                top_n = st.slider("Number of top authors (After)", min_value=5, max_value=50, value=15, key="after_author_slider")
                chart = create_author_collaboration_network(st.session_state.df_after, top_n=top_n)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("Insufficient author data for collaboration network.")
            else:
                st.info("No data available for author analysis.")
    
    # Tab 5: Download Data
    with tabs[4]:
        st.markdown("<h2 class='subheader'>Download Processed Data</h2>", unsafe_allow_html=True)
        
        st.write("""
        The processed data files below have been cleaned and optimized:
        - Verified dates are within appropriate ranges
        - Removed excess empty columns
        - Formatted for easy analysis
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 class='section-header'>Before Approval Data</h3>", unsafe_allow_html=True)
            if st.session_state.excel_before is not None:
                if st.session_state.df_before.empty:
                    st.info("No data available for download.")
                else:
                    st.download_button(
                        label="Download Excel for Before Approval Articles",
                        data=st.session_state.excel_before,
                        file_name=f"{drug_name}_BeforeApproval.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download the processed data for Before Approval period"
                    )
                    st.write(f"Contains {len(st.session_state.df_before)} articles")
            else:
                st.info("Data not yet processed. Run a search first.")
        
        with col2:
            st.markdown("<h3 class='section-header'>After Approval Data</h3>", unsafe_allow_html=True)
            if st.session_state.excel_after is not None:
                if st.session_state.df_after.empty:
                    st.info("No data available for download.")
                else:
                    st.download_button(
                        label="Download Excel for After Approval Articles",
                        data=st.session_state.excel_after,
                        file_name=f"{drug_name}_AfterApproval.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download the processed data for After Approval period"
                    )
                    st.write(f"Contains {len(st.session_state.df_after)} articles")
            else:
                st.info("Data not yet processed. Run a search first.")

# Footer
st.markdown("""
<div class="footer">
    <p>Developed by Rhenix Life Sciences</p>
    <p>© 2025 Rhenix Life Sciences. All rights reserved.</p>
    <p>Data sourced from PubMed via E-utilities API</p>
</div>
""", unsafe_allow_html=True)