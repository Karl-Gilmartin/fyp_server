package com.urbanroute;
import com.graphhopper.GHRequest;
import com.graphhopper.GHResponse;
import com.graphhopper.GraphHopper;
import com.graphhopper.GraphHopperConfig;
import com.graphhopper.ResponsePath;
import com.graphhopper.config.Profile;
import com.graphhopper.json.Statement;
import com.graphhopper.util.CustomModel;
import com.graphhopper.util.Helper;
import com.graphhopper.util.PointList;
import com.graphhopper.util.shapes.GHPoint;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.ArrayNode;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * 26th April 2025
 */
public class SafetyRoutingCustom {
    private static final String OSM_FILE = "ireland-and-northern-ireland-latest.osm.pbf";
    private static final String SAFETY_WEIGHTS_FILE = "safety_weights.json";
    private static final String GRAPHHOPPER_LOCATION = "graphhopper-data";

    private final GraphHopper graphHopper;
    private final Map<Long, Double> safetyWeights;
    private final ObjectMapper objectMapper;

    public SafetyRoutingCustom() throws IOException {
        this.objectMapper = new ObjectMapper();
        this.safetyWeights = loadSafetyWeights();
        this.graphHopper = createGraphHopper();
    }

    private GraphHopper createGraphHopper() {
        Helper.removeDir(new File(GRAPHHOPPER_LOCATION));

        // Configure GraphHopper
        GraphHopperConfig config = new GraphHopperConfig();
        config.putObject("graph.location", GRAPHHOPPER_LOCATION);
        config.putObject("datareader.file", OSM_FILE);
        config.putObject("graph.encoded_values", "foot_access,foot_average_speed,road_class,road_environment,osm_way_id");
        config.putObject("prepare.min_network_size", 0);
        config.putObject("prepare.min_one_way_network_size", 0);
        
        // Add required import configuration
        config.putObject("import.osm.ignored_highways", Collections.singletonList("motorway"));
        config.putObject("import.osm.process_conditional_tags", false);
        config.putObject("import.osm.allow_other_languages", false);
        config.putObject("import.osm.parse_relations", false);
        
        // Init GraphHopper
        GraphHopper hopper = new GraphHopper();
        hopper.init(config);
        
        // Create a profile for foot routing with custom model
        CustomModel customModel = new CustomModel();
        customModel.addToSpeed(Statement.If("true", Statement.Op.LIMIT, "5")); // Set base speed to 5 km/h
        Profile profile = new Profile("foot")
            .setWeighting("custom")
            .setCustomModel(customModel);
        
        hopper.setProfiles(Collections.singletonList(profile));
        hopper.importOrLoad();
        
        return hopper;
    }

    private Map<Long, Double> loadSafetyWeights() {
        Map<Long, Double> weights = new HashMap<>();
        try {
            JsonNode root = objectMapper.readTree(new File(SAFETY_WEIGHTS_FILE));
            JsonNode unsafeWays = root.get("unsafe_ways");
            if (unsafeWays != null && unsafeWays.isArray()) {
                for (JsonNode way : unsafeWays) {
                    long wayId = way.get("way_id").asLong();
                    double weight = way.get("weight").asDouble();
                    String description = way.get("description").asText();
                    weights.put(wayId, weight);
                    System.out.println("Loaded unsafe way: " + description + " (ID: " + wayId + ", Weight: " + weight + ")");
                }
            }
        } catch (IOException e) {
            System.err.println("Error loading safety weights: " + e.getMessage());
            weights.put(1288829638L, 25.0);
            weights.put(556390402L, 15.0);
            weights.put(3791747L, 30.0);
            weights.put(282571104L, 20.0);
            weights.put(50214631L, 30.0);
            weights.put(12341247L, 25.0);
        }
        return weights;
    }

    public void calculateRoute(double fromLat, double fromLon, double toLat, double toLon, boolean useSafetyWeights) {
        GHPoint from = new GHPoint(fromLat, fromLon);
        GHPoint to = new GHPoint(toLat, toLon);

        CustomModel customModel = new CustomModel();
        
        if (useSafetyWeights) {
            // Apply penalties for unsafe ways based on their weights
            for (Map.Entry<Long, Double> entry : safetyWeights.entrySet()) {
                long wayId = entry.getKey();
                double weight = entry.getValue();
                
                // This needs to be updated / set by the user
                // Calculate penalty based on weight (higher weight = more unsafe = higher penalty)
                double penalty = Math.max(0.1, 1.0 - (weight / 50.0)); // minimum speed is 10%
                
                // Add a condition for this specific way ID
                String condition = "osm_way_id == " + wayId;
                customModel.addToSpeed(Statement.If(condition, Statement.Op.MULTIPLY, String.format("%.2f", penalty)));
                
                System.out.println("Added penalty for way " + wayId + ": " + penalty);
            }
        } else {
            // No safety weights, just use base speed
            customModel.addToSpeed(Statement.If("true", Statement.Op.LIMIT, "5"));
        }

        // Calculate the route using GraphHopper's routing
        GHRequest request = new GHRequest(from, to)
            .setProfile("foot")
            .setCustomModel(customModel);
        
        // Add routing hints to encourage alternative path finding
        request.putHint("ch.disable", true); // Disable speed mode when using custom model
        request.putHint("routing.max_visited_nodes", "10000");
        request.putHint("routing.alternative_route.max_paths", "5");
        request.putHint("routing.alternative_route.max_weight_factor", "3.0");
        request.putHint("routing.alternative_route.min_plateau_factor", "0.2");
        request.putHint("routing.alternative_route.max_exploration_factor", "5");

        GHResponse response = graphHopper.route(request);
        if (response.hasErrors()) {
            System.out.println("Error calculating route: " + response.getErrors());
            return;
        }

        ResponsePath path = response.getBest();
        if (path == null) {
            System.out.println("No route found!");
            return;
        }

        System.out.println("Route " + (useSafetyWeights ? "with" : "without") + " safety weights:");
        System.out.println("Distance: " + String.format("%.2f", path.getDistance()) + "m");
        System.out.println("Time: " + path.getTime() / 1000 + "s");

        // Save the route to a GeoJSON file
        String filename = "route_" + (useSafetyWeights ? "with" : "without") + "_safety-custom.geojson";
        saveRouteAsGeoJson(path.getPoints(), filename);
    }

    private void saveRouteAsGeoJson(PointList points, String filename) {
        try {
            ObjectNode featureCollection = objectMapper.createObjectNode();
            featureCollection.put("type", "FeatureCollection");
            ArrayNode features = objectMapper.createArrayNode();

            // Create a LineString feature for the route
            ObjectNode feature = objectMapper.createObjectNode();
            feature.put("type", "Feature");
            
            ObjectNode geometry = objectMapper.createObjectNode();
            geometry.put("type", "LineString");
            ArrayNode coordinates = objectMapper.createArrayNode();

            // Add all points from the path
            for (int i = 0; i < points.size(); i++) {
                ArrayNode coord = objectMapper.createArrayNode();
                coord.add(points.getLon(i));
                coord.add(points.getLat(i));
                coordinates.add(coord);
            }

            geometry.set("coordinates", coordinates);
            feature.set("geometry", geometry);
            features.add(feature);
            featureCollection.set("features", features);

            // Write to file
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(new File(filename), featureCollection);
            System.out.println("Route saved to " + filename);
        } catch (IOException e) {
            System.err.println("Error saving route to GeoJSON: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        try {
            SafetyRoutingCustom router = new SafetyRoutingCustom();
            
            // Example coordinates (Dublin city center)
            double fromLat = 53.343;
            double fromLon = -6.27;
            double toLat = 53.355;
            double toLon = -6.25;

            // Calculate route without safety weights
            router.calculateRoute(fromLat, fromLon, toLat, toLon, false);

            // Calculate route with safety weights
            router.calculateRoute(fromLat, fromLon, toLat, toLon, true);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
} 