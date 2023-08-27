# AUTOCDR

### WORK IN PROGRESS!!!!!!!!

This comprehensive guide provides detailed insights into the **Autocdr App**. Built using Flask, a Python web framework, the app incorporates various libraries and tools to process and analyze retinal images. The app is hosted on a GPU-enabled Docker container orchestrated by Kubernetes for optimized performance.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [App Overview](#app-overview)
4. [Usage](#usage)
5. [AWS Configuration](#aws-configuration)
6. [Models](#models)
7. [Image Processing and Analysis](#image-processing-and-analysis)
8. [Deployment](#deployment)
9. [Endpoints](#endpoints)
10. [Running the App](#running-the-app)
11. [Contact](#contact)

## Introduction

The **Retinal Image Analysis App** is designed to process retinal images and perform analysis to calculate the Cup-Disc Ratio (CDR) – a crucial measurement in the diagnosis of conditions like glaucoma. The app employs pre-trained machine learning models to predict the regions of interest in retinal images, while providing insightful visualizations of the analysis results.

## Setup

1. **Clone the Repository:** Clone or download the repository containing the app code and related files.

2. **Install Dependencies:** Ensure you have Python 3.x installed. Navigate to the app's directory and run the following command to install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. **Configuration:** Open the `app.py` file and configure the AWS access key, secret key, and other settings according to your needs. Additionally, ensure the pre-trained model files are placed in the appropriate folders, as specified in the code.

## App Overview

The app is built using Flask, a lightweight and efficient web framework. It combines HTML templates, CSS, and Python code to create a user-friendly interface for uploading retinal images and visualizing the CDR analysis results.

## Usage

1. **Upload Image:** Open the app in a web browser. The home page enables users to upload retinal images for analysis. Select an image file using the provided upload button and click "Submit."

2. **Processing and Analysis:** Upon uploading an image, the app undertakes the following steps:
    - Reads the image data and resizes it to 512x512 pixels.
    - Utilizes pre-trained machine learning models to predict the cup and disc regions within the image.
    - Calculates the Cup-Disc Ratio (CDR) using the predictions.
    - Generates visualizations, such as overlay images and CDR contour plots, to facilitate result comprehension.

3. **Results:** The app displays the computed CDR values alongside visualization components illustrating the analysis results. Users can view overlay images, contour plots, and more.

## AWS Configuration

The app integrates with Amazon Web Services (AWS) S3 to facilitate image upload and retrieval. Users should possess an AWS account and the requisite credentials. AWS-related settings, including the access key, secret key, and bucket name, are configured in the `app.py` file.

## Models

Pre-trained deep learning models are central to the app's functioning, as they predict the cup and disc regions in retinal images. These models are loaded from specified paths and employed to make accurate predictions.

## Image Processing and Analysis

The core of the app lies in its image processing and analysis steps. Uploaded retinal images are processed using the loaded models to predict cup and disc regions. The Cup-Disc Ratio (CDR) is subsequently calculated based on these predictions. The app generates visualizations to assist in understanding the analysis results.

## Deployment

The Retinal Image Analysis App is deployed within a GPU-enabled Docker container. Kubernetes is employed for orchestration, facilitating efficient resource utilization and scaling. This setup ensures optimal performance for the demanding image processing tasks.

## Endpoints

The app offers two main endpoints:

- **`/`**: The home page enables users to upload retinal images for analysis.
- **`/upload`**: This endpoint processes uploaded images, performs analysis, and displays results.

## Running the App

To run the app locally, navigate to the app's directory and execute the following command:

```bash
python app.py
```



# Deployment Instructions

## Running with Docker

The app can also be deployed using Docker for consistent environment management. Follow these steps to run the app using Docker:

1. **Build the Docker Image:** In the app's directory, run the following command to build the Docker image:

   ```bash
   docker build -t retinal-image-app .
```

Run the Docker Container: Once the image is built, start a Docker container using the following command:

```bash
docker run -p 5000:5000 retinal-image-app
``````
Access the App: The app will be accessible in a web browser by navigating to http://localhost:5000.



## Running on Kubernetes
To deploy the app on a Kubernetes cluster, follow these steps:

Set Up a Kubernetes Cluster: You have a couple of options for setting up a Kubernetes cluster:

Using eksctl: If you prefer an easy way to set up a managed Kubernetes cluster on AWS, you can use eksctl. Install eksctl and run the following command:

bash
Copy code
eksctl create cluster --name my-kubernetes-cluster --region us-east-1
Using kops: Alternatively, if you need more customization, you can use kops to create a Kubernetes cluster on AWS. Install kops and follow the documentation to set up your cluster.

Domain Registration: Register a domain and configure it to point to your Kubernetes cluster. This is necessary to access the app using a custom domain.

Deploy Using Kubernetes YAML: Use the provided Kubernetes YAML file to deploy the app:

bash
Copy code
kubectl apply -f app-deployment.yaml
Replace YOUR_DOMAIN with your registered domain.

Access the App: After the deployment, the app will be accessible using your custom domain, such as http://your-domain.com.




This will initiate a local development server. The app can be accessed by opening a web browser and entering http://localhost:5000 in the address bar.

Contact
For any inquiries or assistance regarding the app, please don't hesitate to contact the developer at iaanimashaun@gmail.com

