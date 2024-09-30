import sys
import os
import pandas as pd  # Import pandas
from tqdm import tqdm  # Import tqdm for progress bar
import pymupdf4llm as plm
from concurrent.futures import ProcessPoolExecutor

from ReadPDF import process_pdf
from toc import extract_toc
from graphio import save_gml, read_gml

from typing import Tuple
import pickle
import ollama
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import re
from tqdm import tqdm
import time
import json
import dspy
from typing import List, Union, Optional
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from rake_nltk import Rake

# Ensure you have the necessary NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')



os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = "cugraph"