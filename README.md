# Building and Floor Management System

## Overview

This project is a comprehensive web application for managing buildings and their associated floors, integrating spatial data, computer vision, cloud storage, and modern frontend technologies. It leverages AI libraries, MongoDB, Cloudinary, and the Nominatim API for seamless data handling and visualization, demonstrating practical applications of Big Data concepts.

## Features

**Building and Floor Management:**

*   CRUD operations for buildings and floors with real-time data rendering.
    
*   Upload and manage floor plans and associated geographical data.
    

**AI and Computer Vision Integration:**

*   Extract metadata and features from uploaded images using AI libraries like OpenCV and TensorFlow.
    
*   Automate GeoJSON generation for floor mapping.
    

**Geocoding with Nominatim API:**

*   Automatically fetch the bounding box coordinates of a building address.
    
*   Enhance location data accuracy for mapping purposes.
    

**Cloud Storage:**

*   Cloudinary for secure and scalable storage of images and files.
    

**Big Data System Concepts:**

*   MongoDB for document-based data storage.
    
*   GeoJSON for spatial data management.
    
*   APIs for efficient data processing and visualization.
    

**Interactive Maps:**

*   Mapbox for visualizing building and floor spatial data.
    
*   Update and download spatial data in GeoJSON format.
    

## Technical Stack

**Backend**

*   Framework: FastAPI
    
*   Database: MongoDB
    
*   Storage: Cloudinary
    
*   Geocoding: Nominatim API for address-to-coordinate conversion.
    

**Frontend**

*   Framework: React.js
    
*   Styling: Tailwind CSS
    

**APIs**

*   **Custom APIs:**
    
    *   CRUD operations for buildings and floors.
        
    *   Image processing and GeoJSON generation.
        
*   **Third-party APIs:**
    
    *   Cloudinary: For image handling.
        
    *   Mapbox: For spatial data visualization.
        
    *   Nominatim: For geocoding.
        

## Key Backend Features

**Building and Floor Management:**

*   **API Endpoints:**
    
    *   `POST /buildings`: Add a building.
        
    *   `GET /buildings`: Retrieve all buildings.
        
    *   `POST /buildings/:id/floors`: Add floors to a building.
        
    *   `GET /buildings/:id/floors`: Retrieve floors for a building.
        

**AI and Computer Vision:**

*   AI libraries preprocess floor plans to extract spatial data and generate GeoJSON files.
    

**Geocoding with Nominatim API:**

*   Converts building addresses into bounding box coordinates for precise spatial visualization.
    

**MongoDB:**

*   Stores structured building and floor data efficiently.
    
*   **Example Document:**
    
    ```json
    {
      "buildingId": "unique-id",
      "name": "Building Name",
      "address": "Building Address",
      "coordinates": [longitude, latitude],
      "floors": [
        {
          "floorId": "unique-id",
          "name": "Floor Name",
          "geojson": {...},
          "imageUrl": "path-to-cloudinary-image"
        }
      ]
    }
    ```

**Cloudinary:**

*   Handles image storage and delivery.
    
*   Optimizes images for size and performance dynamically.
    

**GeoJSON:**

*   Manages and exports spatial data for floors.
    

## Frontend Highlights

*   Dynamic Routing: React Router enables seamless navigation between buildings and floors.
    
*   Interactive Maps: Mapbox visualizes spatial data and integrates with GeoJSON files.
    

## Setup Instructions

**Backend**

1.  **Set Up a Virtual Environment:**
    
    ```bash
    python -m venv env
    source env/bin/activate  # For MacOS/Linux
    env\Scripts\activate     # For Windows
    ```
    
2.  **Install Dependencies:**
    
    ```bash
    pip install -r requirements.txt
    ```
    
3.  **Set Up Environment Variables:**
    
    Create a `.env` file in the root directory and add:
    
    ```env
    MONGO_URI="your_mongodb_uri"
    CLOUDINARY_CLOUD_NAME="your_cloudinary_name"
    CLOUDINARY_API_KEY="your_cloudinary_key"
    CLOUDINARY_API_SECRET="your_cloudinary_secret_key"
    ```
    
4.  **Run the Server:**
    
    ```bash
    uvicorn main:app --reload
    ```

**Frontend**

1.  **Install Dependencies:**
    
    ```bash
    npm install
    ```
    
2.  **Start the Development Server:**
    
    ```bash
    npm start
    ```
    
3.  **Access the App:**
    
    Navigate to `http://localhost:3000`.
    

## Big Data System Concepts

*   **Distributed Data Storage:** MongoDB and Cloudinary manage large-scale data efficiently.
    
*   **Spatial Data:** GeoJSON and Nominatim API enhance geospatial querying and visualization.
    
*   **Data Processing Pipelines:** AI-based image processing simulates Big Data pipelines for automated analysis.
    

## Future Enhancements

*   **Advanced Analytics:** Add dashboards for building and floor usage analytics.
    
*   **3D Mapping:** Introduce 3D floor and building visualizations using Mapbox.
    
*   **User Management:** Role-based access control for secure collaboration.
