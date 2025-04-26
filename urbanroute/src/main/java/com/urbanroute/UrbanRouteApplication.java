package com.urbanroute;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Scope;

@SpringBootApplication
public class UrbanRouteApplication {

    public static void main(String[] args) {
        SpringApplication.run(UrbanRouteApplication.class, args);
    }

    @Bean
    @Scope("singleton")
    public SafetyRoutingCustom safetyRoutingCustom() throws Exception {
        return new SafetyRoutingCustom();
    }
} 