# Building and Floor Management System

## Overview

This project is a comprehensive web application for managing buildings and their associated floors, integrating spatial data, computer vision, cloud storage and modern frontend technologies. It leverages AI libraries, MongoDB, Cloudinary, and the Nominatim API for seamless data handling and visualization, demonstrating practical applications of Big Data concepts.

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
    

## Features

The frontend of the application is developed using React.js and styled with Tailwind CSS, offering a clean and responsive user interface. Below are the key features of the frontend, explained in sequence, corresponding to the 10 provided images:

1.  **Buildings Overview Page:** Displays all the buildings stored in the system with options to view, edit, or delete each building.
    
2.  **Adding a New Building:** Provides a user-friendly form to add a new building with a name and address.
    
3.  **Building Details and Floor Management:** Shows detailed information about a selected building, including a list of its associated floors.
    
4.  **Adding a New Floor:** Enables users to add a new floor to a building by entering its name and floor number.
    
5.  **Floor Details with Map Integration:** Displays floor-specific details along with a Mapbox-based visualization for spatial mapping.
    
6.  **Uploading Floor Plans:** Allows users to upload floor plan images. The uploaded images are processed and securely stored in Cloudinary.
    
7.  **View Floor Plans and GeoJSON Options:** Offers a side-by-side display of the original floor plan, cropped floor plan, and a GeoJSON-based coordinate map.
    
8.  **Copy or Download GeoJSON Data:** Enables users to copy the GeoJSON data to the clipboard or download it as a zip file for external use.
    
9.  **Editing Coordinates and Floor Data:** Provides interactive tools to edit floor coordinates and align the spatial data on the map.
    
10.  **Finalized Floor and Building Overview with Data Visualization:** Summarizes the building and floor data with comprehensive visualizations and exportable GeoJSON files.
    

This workflow ensures a seamless and interactive user experience, fully integrating the frontend with the backend functionality.

## Key Functional Areas

**Building and Floor Management:**

*   Perform CRUD operations for buildings and floors with real-time data updates.
    
*   Upload and manage floor plans alongside their geographical metadata.
    

**AI and Computer Vision Integration:**

*   Extract metadata and features from uploaded floor plan images using AI libraries such as OpenCV and TensorFlow.
    
*   Automate the generation of GeoJSON files for floor mapping.
    

**Geocoding with Nominatim API:**

*   Automatically retrieve bounding box coordinates for a building’s address.
    
*   Enhance the accuracy of location-based visualizations.
    

**Cloud Storage:**

*   Use Cloudinary for secure, scalable image storage and transformations.
    

**Big Data System Concepts:**

*   Leverage MongoDB for efficient and scalable document-based data storage.
    
*   Utilize GeoJSON for structured spatial data management.
    
*   Implement APIs to support robust and efficient data processing workflows.
    

**Interactive Maps:**

*   Integrate Mapbox to visualize building and floor spatial data dynamically.
    
*   Provide options to update, edit, and download GeoJSON data directly from the map interface.
    

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
  
    
 ## Images and Application Flow

I have included a folder in the GitHub repository with 10 numbered images demonstrating the application's functionality. These images represent the order in which the app is used:

*   Image 1: Buildings Overview Page
    
*   Image 2: Adding a New Building
    
*   Image 3: New Added Building Shown on Page
    
*   Image 4: New Building Added Page
    
*   Image 5: Added New Floor to the Building
    
*   Image 6: Upload Floor Plan of the Floor
    
*   Image 7: GeoJSON Visualization of Floor Plan on Map
    
*   Image 8: Can See Cropped Floor Plan (Processing Done at the Backend)
    
*   Image 9: Processed Floor Plan Image Used to Get the GeoJSON Data
    
*   Image 10: Can Adjust Coordinates to Fit the Floor Plan at the Right Place

  
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


