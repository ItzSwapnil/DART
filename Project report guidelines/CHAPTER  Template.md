

# CHAPTER 1: INTRODUCTION

The design of a drone-based agricultural surveillance system aims to optimize resource utilization, enhance crop yields, and minimize environmental impacts through advanced technology integration. This system leverages various components such as drones, ground control stations (GCS), and sophisticated software to facilitate efficient agricultural monitoring and management.

## 1.1 Drone Communication and Control

### 1.1.1 Data Links

Drones utilized in agricultural surveillance typically operate on stable communication frequencies, including 2.4 GHz and 5.8 GHz, ensuring uninterrupted control and data transfer during missions [12]. The integration of 4G and 5G networks further extends the operational range of these drones, enabling Beyond Visual Line of Sight (BVLOS) operations, which is crucial for large-scale surveillance across expansive farmland [12].

### 1.1.2 Ground Control Stations (GCS)

Ground control stations play a pivotal role in monitoring drone operations and managing mission plans. Modern GCS are equipped with powerful computing resources, high-resolution displays, and user-friendly interfaces that facilitate real-time mission management. The accompanying software allows for detailed mission planning, optimizing drone routes, and incorporating various data inputs, such as weather conditions and regulatory guidelines, to enhance operational efficiency [12].

### 1.1.3 Drone Payloads

The payload capabilities of agricultural drones vary by model, impacting their utility across different farm sizes and applications. For example, the Agras T70P drone can carry significant payloads—up to 70 liters for spraying or 100 liters for spreading—while also integrating advanced safety systems to enhance obstacle detection and route planning [13][14]. In contrast, the compact Agras T25P is tailored for individual operators and small-scale farms, featuring a high-precision spreading system and fully automated functions for various agricultural tasks [14].

## 1.2 Surveillance Capabilities

### 1.2.1 Imaging Technology

Drones equipped with high-resolution cameras, thermal sensors, and LiDAR technology enable comprehensive surveillance of agricultural fields. These technologies facilitate the collection of vital data regarding crop health, soil conditions, and environmental changes. By accessing hard-to-reach areas and capturing images from elevated positions, drones offer a unique perspective that enhances decision-making for farmers and agronomists [15].

### 1.2.2 Automation and Data Processing

The effective operation of a drone surveillance system requires the integration of intelligent automation software, which allows for pre-programmed flight paths and real-time data monitoring. The automation capabilities include functions like aerial mapping and terrain-following spraying, ensuring precision and efficiency in agricultural practices. Data collected during flights can be analyzed to inform farming strategies and improve yield outcomes [15][16].

![System Architecture: Drone-Based Agricultural Surveillance. The diagram is divided into four horizontal layers. Top layer: AERIAL PLATFORMS, showing AGRAS T70P (Spraying/Spreading) with LIDAR and VIDEO THERMAL SENSORS, and AGRAS T25P (Individual Farms) with CHIRP-8 RADAR and HIGH-RES CAMERA. Middle layer: COMMUNICATION INFRASTRUCTURE, showing 2.4/5.8 GHz RADIO and 4G/5G NETWORKS, with BVLOS CAPABILITIES (Long Range Data & Control). Bottom layer: COMMAND & CONTROL, showing GROUND CONTROL STATION (GCS) with USER-FRIENDLY INTERFACE and FLIGHT SOFTWARE including MISSION PLANNING, ROUTE OPTIMIZATION, and REGULATORY GUIDELINES. Bottom-most layer: DATA & ANALYTICS, showing CLOUD/LOCAL SERVER, INTELLIGENT AUTOMATION SOFTWARE (REAL-TIME MONITORING), AERIAL MAPPING / NDVI, YIELD OPTIMIZATION, and SOIL HEALTH ANALYTICS.](4792a2ccd62226861fadc22117edb7b1_img.jpg)

System Architecture: Drone-Based Agricultural Surveillance. The diagram is divided into four horizontal layers. Top layer: AERIAL PLATFORMS, showing AGRAS T70P (Spraying/Spreading) with LIDAR and VIDEO THERMAL SENSORS, and AGRAS T25P (Individual Farms) with CHIRP-8 RADAR and HIGH-RES CAMERA. Middle layer: COMMUNICATION INFRASTRUCTURE, showing 2.4/5.8 GHz RADIO and 4G/5G NETWORKS, with BVLOS CAPABILITIES (Long Range Data & Control). Bottom layer: COMMAND & CONTROL, showing GROUND CONTROL STATION (GCS) with USER-FRIENDLY INTERFACE and FLIGHT SOFTWARE including MISSION PLANNING, ROUTE OPTIMIZATION, and REGULATORY GUIDELINES. Bottom-most layer: DATA & ANALYTICS, showing CLOUD/LOCAL SERVER, INTELLIGENT AUTOMATION SOFTWARE (REAL-TIME MONITORING), AERIAL MAPPING / NDVI, YIELD OPTIMIZATION, and SOIL HEALTH ANALYTICS.

**Figure 1.** System Architecture

**Table 1.** Comparative Analysis of Drone Technologies and Their Agricultural Applications

| Drone Technology          | Primary Agricultural Use                | Flight Endurance        | Coverage Efficiency | Payload Capacity     |
|---------------------------|-----------------------------------------|-------------------------|---------------------|----------------------|
| <b>Multicopter Drones</b> | Crop monitoring, precision spraying     | Low–Medium (15–40 min)  | Low–Medium          | Low–Medium (2–10 kg) |
| <b>Fixed-Wing Drones</b>  | Large-scale field mapping and surveying | High (45–120 min)       | High                | Low (1–5 kg)         |
| <b>VTOL Hybrid Drones</b> | Surveying with vertical takeoff         | Medium–High (40–90 min) | Medium–High         | Medium (3–8 kg)      |

|                             |                                      |                          |             |                     |
|-----------------------------|--------------------------------------|--------------------------|-------------|---------------------|
| <b>Spray Drones</b>         | Pesticide and fertilizer application | Low–Medium (20–50 min)   | Medium      | High (10–30 kg)     |
| <b>LiDAR-Enabled Drones</b> | Terrain and crop structure mapping   | Medium–High (40–100 min) | Medium–High | Low–Medium (2–6 kg) |