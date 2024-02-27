from fastapi import FastAPI, File, UploadFile, Form, Request, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.templating import Jinja2Templates
from typing import Annotated
import csv
from io import StringIO
from fastapi.responses import RedirectResponse, HTMLResponse
from bs4 import BeautifulSoup
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = FastAPI()
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

@app.get("/", response_class=FileResponse)
async def read_root():
    return "templates/index.html"

@app.get("/resume", response_class=FileResponse)
async def read_root():
    return "templates/resume/resume.html"

@app.get("/bio", response_class=FileResponse)
async def read_root():
    return "templates/bio/bio.html"

@app.get("/model", response_class=FileResponse)
async def read_root():
    return "templates/readcsv/readcsv.html"

@app.post('/preprocess/{modeltype}', response_class=HTMLResponse)
async def read_csv(modeltype, formFile: UploadFile = File(...)):
    # Check if the uploaded file is a CSV file
    if not formFile.filename.endswith('.csv'):
        return f'{formFile.filename} is not a CSV file.'
    
    # Read the contents of the uploaded CSV file
    contents = await formFile.read()


    # Convert bytes to string
    contents_str = contents.decode()
    
    # Parse CSV data
    csv_data = []
    with StringIO(contents_str) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            csv_data.append(row)
    
    # Encode CSV data as a string
    csv_data_str = '\n'.join([','.join(row) for row in csv_data])

    # Read the contents of the existing HTML file
    with open("templates/preprocess/preprocess.html", "r") as file:
        html_content = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Create a table tag and append rows from CSV data
    table_tag = soup.new_tag('table', **{'class': 'table table-striped'})

    num_rows = 0
    for row in csv_data:
        tr_tag = soup.new_tag('tr')
        num_cols=0
        for cell in row:
            td_tag = soup.new_tag('td')
            if num_cols>5:
                td_tag.string = '...'
                tr_tag.append(td_tag)
            else:
                td_tag.string = cell
                tr_tag.append(td_tag)
            num_cols+=1
        table_tag.append(tr_tag)

        num_rows+=1
        if num_rows>5:
            break

    # Find the body tag and append the table tag
    body_tag = soup.find(id='table')
    body_tag.append(table_tag)

    # accuracy = make_model(modeltype, csv_data)

    # p_tag = soup.new_tag('p')
    # p_tag.string=accuracy
    # body_tag.append(p_tag)
    # Create a dropdown menu

    # Get the modified HTML content
    # modified_html = str(soup)

    # Return the modified HTML content

    # features to remove
    features_rm =[]
    for frm in csv_data[0]:
        print(frm)

    select_frm_tag = soup.new_tag('select', **{'class': 'form-select form-select-lg mb-3'})
    select_frm_tag['onchange'] = 'getRMFeatues(this)'
    num_rows = 0
    for row in csv_data[0]:
        option_tag = soup.new_tag('option')
        option_tag.string = row
        select_frm_tag.append(option_tag) 

    body_tag = soup.find(id='select-frm')
    body_tag.append(select_frm_tag) 

    modified_html = str(soup)

    return HTMLResponse(content=modified_html)

@app.get("/preprocess", response_class=FileResponse)
async def read_root():
    return "templates/preprocess/preprocess.html"

@app.post('/model/{modeltype}', response_class=HTMLResponse)
async def read_csv(modeltype, formFile: UploadFile = File(...)):
    # Check if the uploaded file is a CSV file
    if not formFile.filename.endswith('.csv'):
        return f'{formFile.filename} is not a CSV file.'
    
    # Read the contents of the uploaded CSV file
    contents = await formFile.read()


    # Convert bytes to string
    contents_str = contents.decode()
    
    # Parse CSV data
    csv_data = []
    with StringIO(contents_str) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            csv_data.append(row)


    
    # Encode CSV data as a string
    csv_data_str = '\n'.join([','.join(row) for row in csv_data])

    # Read the contents of the existing HTML file
    with open("templates/upload/upload.html", "r") as file:
        html_content = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Create a table tag and append rows from CSV data
    table_tag = soup.new_tag('table', **{'class': 'table table-striped'})

    num_rows = 0
    for row in csv_data:
        tr_tag = soup.new_tag('tr')
        num_cols=0
        for cell in row:
            td_tag = soup.new_tag('td')
            if num_cols>5:
                td_tag.string = '...'
                tr_tag.append(td_tag)
            else:
                td_tag.string = cell
                tr_tag.append(td_tag)
            num_cols+=1
        table_tag.append(tr_tag)

        num_rows+=1
        if num_rows>5:
            break

    # Find the body tag and append the table tag
    body_tag = soup.find(id='table')
    body_tag.append(table_tag)

    accuracy = make_model(modeltype, csv_data)

    p_tag = soup.new_tag('p')
    p_tag.string=accuracy
    body_tag.append(p_tag)
    # Create a dropdown menu

    # Get the modified HTML content
    modified_html = str(soup)

    # Return the modified HTML content
    
    return HTMLResponse(content=modified_html)

@app.post('/serving/{modeltype}', response_class=HTMLResponse)
async def read_csv(modeltype, formFile: UploadFile = File(...)):
    # Check if the uploaded file is a CSV file
    if not formFile.filename.endswith('.csv'):
        return f'{formFile.filename} is not a CSV file.'
    
    # Read the contents of the uploaded CSV file
    contents = await formFile.read()


    # Convert bytes to string
    contents_str = contents.decode()
    
    # Parse CSV data
    csv_data = []
    with StringIO(contents_str) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            csv_data.append(row)


    
    # Encode CSV data as a string
    csv_data_str = '\n'.join([','.join(row) for row in csv_data])

    # Read the contents of the existing HTML file
    with open("templates/upload/upload.html", "r") as file:
        html_content = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Create a table tag and append rows from CSV data
    table_tag = soup.new_tag('table', **{'class': 'table table-striped'})

    num_rows = 0
    for row in csv_data:
        tr_tag = soup.new_tag('tr')
        num_cols=0
        for cell in row:
            td_tag = soup.new_tag('td')
            if num_cols>5:
                td_tag.string = '...'
                tr_tag.append(td_tag)
            else:
                td_tag.string = cell
                tr_tag.append(td_tag)
            num_cols+=1
        table_tag.append(tr_tag)

        num_rows+=1
        if num_rows>5:
            break

    # Find the body tag and append the table tag
    body_tag = soup.find(id='table')
    body_tag.append(table_tag)

    accuracy = make_model(modeltype, csv_data)

    p_tag = soup.new_tag('p')
    p_tag.string=accuracy
    body_tag.append(p_tag)
    # Create a dropdown menu

    # Get the modified HTML content
    modified_html = str(soup)

    # Return the modified HTML content
    
    return HTMLResponse(content=modified_html)



def make_model(modeltype, csv_data):

    dataset = csv_data
    columns_name =  dataset.pop(0)
    # dataset.pop(0)
    df = pd.DataFrame(dataset, columns =columns_name)

    # Initialize linear regression model
    if modeltype=="regression":
        X = df.iloc[:, :-1].astype(float)  # Features (all columns except the last one)
        y = df.iloc[:, -1].astype(float)   # Target variable (last column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        # Train the model
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        # print("Mean Squared Error (MSE):", mse)

        # Calculate correlation coefficient
        correlation = np.corrcoef(y_test, y_pred)[0, 1]
        # print("Correlation coefficient:", correlation)

        return "Correlation coefficient: "+str(correlation)
    elif modeltype=="category":
        X = df.iloc[:, :-1]  # Features (all columns except the last one)
        y = df.iloc[:, -1].astype(object)   # Target variable (last column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize decision tree classifier
        clf = DecisionTreeClassifier()

        # Train the classifier
        clf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        return "accuracy: "+str(accuracy)