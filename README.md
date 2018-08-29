# Ros_workspace


# Package - turtle_draw

### Instructions to Run the Package:
In terminal Use:
<code>$ roslaunch turtle_draw myturtle.launch</code>

This will open the turtle sim and after 2 seconds the turtle performs a motion.
<br />This below pattern will be observed:


![Task1.png](attachment:Task1.png)


# Package : learning_tf

The package contains node to mimic the turtle1.

### Instructions to run the package.
Run the below command :
<code>$ roslaunch learning_tf start_demo.launch </code>

- Now use your keyboard to control the turtle1.(ensure that the focus of mouse in on the terminal from you launched it)<br/>

- You should notice the turtle 2 mimic the turtle 1

# Package : car_gazebo    car_description
<br />
This package contains the nodes and files required to simulate a car in gazebo environment. It also contains files to do a lane detection.
<br />
## Instructions to run :
Run the below command in the terminal: <code>$ roslaunch car_gazebo myworld.launch</code>

The following things will be opened.
- Gazebo
- RVIZ
- Opencv Image window

### Instructions to control the car
- Click on the gazebo window.
- Use "w" key to move the car forward.
- Use "s" key to move backward
- Press "q" or "e" to break.

### Lane Detection
When you move straight you will be able to notice to the lines being detected. It even tries to identify if it is straight road or a turn comming up.


## The Package uses following things for its funtionality.
- Ros (any version after Indigo)
- Ros Gazebo Packages.
- Gazebo 9
- Ignition libraries (usually available with gazebo8) -

