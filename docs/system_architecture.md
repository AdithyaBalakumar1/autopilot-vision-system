# System Architecture

## Overview
This project implements a vision-based autonomous driving research pipeline inspired by publicly discussed Autopilot-style systems.

The system processes camera input to perform perception, planning, and control in a simulated environment.

## Design Principles
- Vision-only input (camera-based)
- Modular architecture with clear separation of concerns
- Explainable, non–black-box components
- Educational and research-focused implementation

## High-Level Pipeline
Camera Input → Perception → Planning → Control → Simulated Output

## Module Responsibilities

### Perception Module
The perception module is responsible for understanding the driving scene from visual input.

Functions:
- Lane detection
- Vehicle detection (future extension)

Input:
- RGB image or video frame

Output:
- Lane boundaries
- Detected objects (bounding boxes, class labels)

---

### Planning Module
The planning module decides driving behavior based on perception outputs.

Functions:
- Lane-following logic
- Basic obstacle avoidance rules

Input:
- Lane and object information from perception

Output:
- Desired steering direction or trajectory

---

### Control Module
The control module converts planning decisions into low-level control commands.

Functions:
- Steering command generation
- Simulation-only control logic

Input:
- Desired steering direction

Output:
- Steering angle (simulated)

## Constraints and Scope
- No real-world vehicle deployment
- No real-time performance guarantees
- No proprietary or private datasets
- No affiliation with Tesla or any automotive company

## Future Extensions
- Traffic sign recognition
- Multi-camera perception
- Simulator integration (e.g., CARLA)
- Learning-based planning models
