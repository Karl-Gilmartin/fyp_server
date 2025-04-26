# Urban Route Safety Routing

This project implements safety-based routing using GraphHopper, allowing users to calculate routes that avoid unsafe areas based on custom safety weights.

## Setup Instructions

### Prerequisites

- Java 11 or higher
- Maven 3.6 or higher

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd urbanroutev3
   ```

2. Download the OSM data file:
   ```
   wget -O ireland-and-northern-ireland-latest.osm.pbf https://download.geofabrik.de/europe/ireland-and-northern-ireland-latest.osm.pbf

   ```
   
3. Build the project:
   ```
   mvn clean install
   ```

4. Run the application:
   ```
   mvn exec:java -Dexec.mainClass="com.urbanv3.SafetyRoutingCustom"
   ```

## Project Structure

- `src/main/java/com/urbanv3/` - Java source files
- `safety_weights.json` - Configuration file for safety weights
- `route_with_safety-custom.geojson` - Generated route with safety considerations
- `route_without_safety-custom.geojson` - Generated route without safety considerations
- `safety_routes_map_custom.html` - HTML visualization of the routes

## Safety Weights

The safety weights are defined in `safety_weights.json`. Each entry contains:
- `way_id`: The OSM way ID
- `weight`: A value indicating how unsafe the way is (higher values = more unsafe)
- `description`: A description of why the way is considered unsafe

## Visualization

To view the routes on a map, open `safety_routes_map_custom.html` in a web browser. The map shows:
- Standard route (blue)
- Safety route (red)
- Custom safety route (green)

You can toggle the visibility of each route using the buttons at the top of the page. 