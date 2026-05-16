C4Component
title Image Behaviour Alerts - Component Diagram

Person(operator, "Operator")

System_Ext(rtsp, "RTSP Camera")
System_Ext(yolo, "YOLO Runtime")

Container_Boundary(app, "Python Desktop Application") {

    Component(ui, "Desktop UI", "Tkinter")

    Component(capture, "Frame Capture", "OpenCV")
    Component(monitor, "Behaviour Monitoring", "Monitoring Loop")
    Component(detector, "Object Detection", "YOLO Wrapper")
    Component(alerts, "Alert Notifications", "Console + Beep")
    Component(preview, "Image Preview", "Tk PhotoImage")
}

Rel(operator, ui, "Starts capture and monitoring")

Rel(ui, capture, "Captures one frame")
Rel(ui, monitor, "Starts monitor")
Rel(ui, preview, "Displays latest frame")

Rel(capture, rtsp, "Reads frames")

Rel(monitor, capture, "Samples frames")
Rel(monitor, detector, "Detects objects")
Rel(monitor, alerts, "Triggers alerts")
Rel(monitor, preview, "Publishes sampled frames")

Rel(detector, yolo, "Runs predictions")

UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
