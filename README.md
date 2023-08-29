# AUTOCDR

### WORK IN PROGRESS!!!!!!!!

This provides detailed insights into the **Autocdr App**. Built using Flask, a Python web framework, the app incorporates various functions to process and analyze retinal images. The app is hosted on a GPU-enabled Docker container orchestrated by Kubernetes for optimized performance.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [App Overview](#app-overview)
4. [Usage](#usage)
5. [Models](#models)
6. [Deployment](#deployment)
7. [Endpoints](#endpoints)
8. [Deployment Instructions](#deployment-instructions)
    <ol style="margin-left: 20px; list-style-type: upper-roman;">
        <li><a href="#running-the-app">Running the App</a></li>
        <li><a href="#running-with-docker">Running with Docker</a></li>
        <li><a href="#running-on-kubernetes">Running on Kubernetes</a></li>
    </ol>
9. [Contact](#contact)



## Introduction

The **Autocdr App** is designed to process retinal images and perform analysis to estimate the Cup-Disc Ratio (CDR) – a crucial measurement in the diagnosis of glaucoma. The app employs pre-trained deep learning models to segment the optic cup and disc in retinal images, while providing insightful visualizations of the results.

## Setup

1. **Clone the Repository:** Clone or download this repository 

2. **Install Dependencies:** Ensure you have Python 3.x installed. Navigate to the app's directory and run the following command to install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. **Configuration:** Download pretrained models and ensure they are placed in the appropriate folders, as specified in the code.

## App Overview

The app is built using Flask, a lightweight and efficient web framework. It combines HTML templates, CSS, and Python code to create a user-friendly interface for uploading retinal images and visualizing the CDR results.

## Usage

1. **Upload Image:** Open the app in a web browser. The home page enables users to upload retinal images for analysis. Select an image file using the provided upload button and click "Submit."

2. **Processing and Analysis:** Upon uploading an image, the app undertakes the following steps:
    - Reads the image data and resizes it to 512x512 pixels.
    - Utilizes pre-trained deep learning models to predict the cup and disc regions within the image.
    - Calculates the Cup-Disc Ratio (CDR) using the predictions.
    - Generates CDR contour plot to facilitate result comprehension.

3. **Results:** The app displays the computed CDR values alongside visualization components illustrating the analysis results. Users can view the CDR and contour plots.

## Models

Pre-trained deep learning models are central to the app's functioning, as they predict the cup and disc regions in retinal images. These models are loaded from specified paths and employed to make accurate predictions.

## Deployment

The Retinal Image Analysis App is deployed within a GPU-enabled Docker container. Kubernetes is employed for orchestration, facilitating efficient resource utilization and scaling. This setup ensures optimal performance for the demanding image processing tasks.

## Endpoints

The app offers two main endpoints:

- **`/`**: The home page enables users to upload retinal images for analysis.
- **`/upload`**: This endpoint processes uploaded images, performs analysis, and displays results.


## Deployment Instructions

### Running the App

To run the app locally, navigate to the app's directory and execute the following command:

```bash
python app.py
```


This will initiate a local development server. The app can be accessed by opening a web browser and entering http://localhost:5000 in the address bar.



### Running with Docker

Follow these steps to run the app using Docker:

1. **Build the Docker Image:** In the app's directory, run the following command to build the Docker image:

```bash
   docker build -t autocdr .
```

2. **Run the Docker Container:** Once the image is built, start a Docker container using the following command:

```bash
docker run -p 5000:5000 autocdr
```

The image is hosted on Dockerhub and one could skip building the image and directly run

```bash
docker run -p 5000:5000 iaanimashaun/autocdr
```

3. **Access the App:** The app will be accessible in a web browser by navigating to http://localhost:5000.



### Running on Kubernetes
To deploy the app on a Kubernetes cluster, follow these steps:

1. **Set Up a Kubernetes Cluster:** You have a couple of options for setting up a Kubernetes cluster:

    **a. Using eksctl:** If you prefer an easy way to set up a managed Kubernetes cluster on AWS, you can use eksctl. Install eksctl and run the following command:

    ```
    eksctl create cluster --name <my-kubernetes-cluster> --region <region-name>
    ```
    **b. Using kops:** Alternatively, if you need more customization, you can use kops to create a Kubernetes cluster on AWS. Install kops and follow the documentation to set up your cluster.

2. **Domain Registration:** Register a domain and configure it to point to your Kubernetes cluster. This is necessary to access the app using a custom domain. One could use Route 53 or any third party domain registrar.

3. **Deploy Using Kubernetes YAML:** Use the provided Kubernetes YAML file to deploy the app:

    ```
    kubectl apply -f autocdr.yaml
    ```
Replace YOUR_DOMAIN.COM with your registered domain.

4. **Access the App:** After the deployment, the app will be accessible using your custom domain, such as http://YOUR_DOMAIN.COM.


## Contact
For any inquiries or assistance regarding the app, please don't hesitate to contact the developer at iaanimashaun@gmail.com

