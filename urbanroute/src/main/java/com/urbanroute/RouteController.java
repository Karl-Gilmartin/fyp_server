package com.urbanroute;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/routes")
@CrossOrigin(origins = "*")
public class RouteController {

    private final SafetyRoutingCustom router;
    private final ObjectMapper objectMapper;

    @Autowired
    public RouteController(SafetyRoutingCustom router, ObjectMapper objectMapper) {
        this.router = router;
        this.objectMapper = objectMapper;
    }

    @GetMapping("/calculate")
    public ResponseEntity<?> calculateRoute(
            @RequestParam double fromLat,
            @RequestParam double fromLon,
            @RequestParam double toLat,
            @RequestParam double toLon,
            @RequestParam(defaultValue = "true") boolean useSafetyWeights) {
        
        try {
            // Calculate the route
            JsonNode routeData = router.calculateRoute(fromLat, fromLon, toLat, toLon, useSafetyWeights);
            
            // Create response with both the GeoJSON and metadata
            Map<String, Object> response = new HashMap<>();
            response.put("route", routeData);
            response.put("metadata", Map.of(
                "from", Map.of("lat", fromLat, "lon", fromLon),
                "to", Map.of("lat", toLat, "lon", toLon),
                "useSafetyWeights", useSafetyWeights // to do: update this to pass in the safety weights or pass in route to safety weights (db path)
            ));
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }
} 