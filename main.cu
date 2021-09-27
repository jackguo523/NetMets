﻿#include <stdlib.h>
#include <string>
#include <fstream>
#include <algorithm> 

#ifdef __CUDACC__
//CUDA includes
#include <cuda.h>
#endif

// STIM includes
#include <stim/parser/arguments.h>
#include <stim/visualization/camera.h>
#include <stim/gl/gl_texture.h>
#include <stim/visualization/gl_network.h>
#include <stim/biomodels/network.h>
#include <stim/visualization/gl_aaboundingbox.h>
#include <stim/visualization/colormap.h>

// OpenGL includes
#include <GL/glut.h>
#include <GL/freeglut.h>


//********************parameter setting*******************
// visualization objects
stim::gl_aaboundingbox<float> bb;	// axis-aligned bounding box object
stim::camera cam;					// camera object

// overall parameters
unsigned num_nets = 0;					// number of networks that've been loaded
float sigma;							// resample rate
float threshold;						// metric acceptable value
float radius = 0.7;						// equals to radius
float delta;							// camera moving parameter
std::vector<float> colormap;			// random generated color set
stim::gl_texture<unsigned char> S;		// texture storing the image stack
float planes[3] = { 0.0f, 0.0f, 0.0f };	// plane position in world space
std::vector<std::string> main_menu_option = { "compare mode", "mapping mode", "volume display"};	// main menu options
std::vector<std::string> sub_menu_option = { "overlaid on", "overlaid off", "highlight on", "highlight off", "light on", "light off" };		// sub menu option
std::string menu_name;
GLuint cmap_tex = 0;					// texture name for the color map
std::vector<unsigned char> brewer_color_map;

// hard-coded parameters
float resample_rate = 0.5f;			// sample rate for the network (fraction of sigma used as the maximum sample rate)
float camera_factor = 1.2f;			// start point of the camera as a function of X and Y size
float orbit_factor = 0.01f;			// degrees per pixel used to orbit the camera
float zoom_factor = 10.0f;			// zooming factor
float radius_factor = 0.5f;			// radius changing factor

// networks
stim::gl_network<float> GT;			// ground truth network
stim::gl_network<float> T;			// test network
stim::gl_network<float> _GT;		// splitted GT
stim::gl_network<float> _T;			// splitted T
std::vector<unsigned> _gt_t;		// store indices of nearest edge in _T for _GT
std::vector<unsigned> _t_gt;		// store indices of nearest edge in _GT for _T

// flags
bool load_image_stack = false;			// flag indicates loading image stacks
bool render_overlaid_network = false;	// flag indicates render a transparant T overlaid on GT
bool open_light = false;				// flag indicates light on/off
bool highlight_difference = false;		// flag indicates highlight the difference between two networks
bool compare_mode = true;				// default mode is compare mode
bool mapping_mode = false;
bool volume_mode = false;

// glut event parameters
bool LButtonDown = false;			// true when left button down
bool RButtonDown = false;
int mouse_x;						// window x-coordinate
int mouse_y;						// window y-coordinate
int mods;							// special keyboard input


//********************parameter setting*******************
// set up the squash transform to whole screen
void glut_render_single_projection() {

	glMatrixMode(GL_PROJECTION);					// load the projection matrix for editing
	glLoadIdentity();								// start with the identity matrix
	int X = glutGet(GLUT_WINDOW_WIDTH);				// use the whole screen for rendering
	int Y = glutGet(GLUT_WINDOW_HEIGHT);
	glViewport(0, 0, X, Y);							// specify a viewport for the entire window
	float aspect = (float)X / (float)Y;				// calculate the aspect ratio
	gluPerspective(60, aspect, 0.1, 1000000);		// set up a perspective projection
}

// set up the squash transform to left half screen
void glut_render_left_projection() {

	glMatrixMode(GL_PROJECTION);					// load the projection matrix for editing
	glLoadIdentity();								// start with the identity matrix
	int X = glutGet(GLUT_WINDOW_WIDTH) / 2;			// only use half of the screen for the viewport
	int Y = glutGet(GLUT_WINDOW_HEIGHT);
	glViewport(0, 0, X, Y);							// specify the viewport on the left
	float aspect = (float)X / (float)Y;				// calculate the aspect ratio
	gluPerspective(60, aspect, 0.1, 1000000);		// set up a perspective projection
}

// set up the squash transform to right half screen
void glut_render_right_projection() {

	glMatrixMode(GL_PROJECTION);					// load the projection matrix for editing
	glLoadIdentity();								// start with the identity matrix
	int X = glutGet(GLUT_WINDOW_WIDTH) / 2;			// only use half of the screen for the viewport
	int Y = glutGet(GLUT_WINDOW_HEIGHT);
	glViewport(X, 0, X, Y);							// specify the viewport on the right
	float aspect = (float)X / (float)Y;				// calculate the aspect ratio
	gluPerspective(60, aspect, 0.1, 1000000);		// set up a perspective projection
}

// translate camera to origin
void glut_render_modelview() {

	glMatrixMode(GL_MODELVIEW);						// load the modelview matrix for editing
	glLoadIdentity();								// start with the identity matrix
	stim::vec3<float> eye = cam.getPosition();		// get the camera position (eye point)
	stim::vec3<float> focus = cam.getLookAt();		// get the camera focal point
	stim::vec3<float> up = cam.getUp();				// get the camera "up" orientation

	gluLookAt(eye[0], eye[1], eye[2], focus[0], focus[1], focus[2], up[0], up[1], up[2]);	// set up the OpenGL camera
}

// three axis slice
// draw x slice
void draw_x_slice(float p) {
	float x = p;
	float y = S.size(1);
	float z = S.size(2);

	float tx = p / S.size(0);		// normalization

	glBegin(GL_QUADS);
	glTexCoord3f(tx, 0, 0);
	glVertex3f(x, 0, 0);

	glTexCoord3f(tx, 0, 1);
	glVertex3f(x, 0, z);

	glTexCoord3f(tx, 1, 1);
	glVertex3f(x, y, z);

	glTexCoord3f(tx, 1, 0);
	glVertex3f(x, y, 0);
	glEnd();
}
// draw y slice
void draw_y_slice(float p) {
	float x = S.size(0);
	float y = p;
	float z = S.size(2);

	float ty = p / S.size(1);

	glBegin(GL_QUADS);
	glTexCoord3f(0, ty, 0);
	glVertex3f(0, y, 0);

	glTexCoord3f(0, ty, 1);
	glVertex3f(0, y, z);

	glTexCoord3f(1, ty, 1);
	glVertex3f(x, y, z);

	glTexCoord3f(1, ty, 0);
	glVertex3f(x, y, 0);
	glEnd();
}
// draw z slice
void draw_z_slice(float p) {
	float x = S.size(0);
	float y = S.size(1);
	float z = p;

	float tz = p / S.size(2);

	glBegin(GL_QUADS);
	glTexCoord3f(0, 0, tz);
	glVertex3f(0, 0, z);

	glTexCoord3f(0, 1, tz);
	glVertex3f(0, y, z);

	glTexCoord3f(1, 1, tz);
	glVertex3f(x, y, z);

	glTexCoord3f(1, 0, tz);
	glVertex3f(x, 0, z);
	glEnd();
}

// draw a bounding box around the data set
void draw_box() {
	float c[3] = { S.size(0), S.size(1), S.size(2) };
	glLineWidth(1.0);

	glBegin(GL_LINE_LOOP);
	glColor3f(0, 0, 0);
	glVertex3f(0, 0, 0);

	glColor3f(0, 1, 0);
	glVertex3f(0, c[1], 0);

	glColor3f(0, 1, 1);
	glVertex3f(0, c[1], c[2]);

	glColor3f(0, 0, 1);
	glVertex3f(0, 0, c[2]);
	glEnd();

	glBegin(GL_LINE_LOOP);
	glColor3f(1, 0, 0);
	glVertex3f(c[0], 0, 0);

	glColor3f(1, 1, 0);
	glVertex3f(c[0], c[1], 0);

	glColor3f(1, 1, 1);
	glVertex3f(c[0], c[1], c[2]);

	glColor3f(1, 0, 1);
	glVertex3f(c[0], 0, c[2]);
	glEnd();

	glBegin(GL_LINES);
	glColor3f(0, 0, 0);
	glVertex3f(0, 0, 0);
	glColor3f(1, 0, 0);
	glVertex3f(c[0], 0, 0);

	glColor3f(0, 1, 0);
	glVertex3f(0, c[1], 0);
	glColor3f(1, 1, 0);
	glVertex3f(c[0], c[1], 0);

	glColor3f(0, 1, 1);
	glVertex3f(0, c[1], c[2]);
	glColor3f(1, 1, 1);
	glVertex3f(c[0], c[1], c[2]);

	glColor3f(0, 0, 1);
	glVertex3f(0, 0, c[2]);
	glColor3f(1, 0, 1);
	glVertex3f(c[0], 0, c[2]);
	glEnd();
}

// draw the plane frame
void draw_frames() {
	float c[3] = { S.size(0), S.size(1), S.size(2) };			// store the size of the data set for all three dimensions

	glLineWidth(1.0);
	glColor3f(1, 0, 0);											// draw the X plane
	glBegin(GL_LINE_LOOP);
	glVertex3f(planes[0], 0, 0);
	glVertex3f(planes[0], c[1], 0);
	glVertex3f(planes[0], c[1], c[2]);
	glVertex3f(planes[0], 0, c[2]);
	glEnd();

	glColor3f(0, 1, 0);											// draw the Y plane
	glBegin(GL_LINE_LOOP);
	glVertex3f(0, planes[1], 0);
	glVertex3f(c[0], planes[1], 0);
	glVertex3f(c[0], planes[1], c[2]);
	glVertex3f(0, planes[1], c[2]);
	glEnd();

	glColor3f(0, 0, 1);											// draw the Z plane
	glBegin(GL_LINE_LOOP);
	glVertex3f(0, 0, planes[2]);
	glVertex3f(c[0], 0, planes[2]);
	glVertex3f(c[0], c[1], planes[2]);
	glVertex3f(0, c[1], planes[2]);
	glEnd();
}

// enforce bound
void enforce_bounds() {
	for (int d = 0; d < 3; d++) {
		if (planes[d] < 0) planes[d] = 0;
		if (planes[d] > S.size(d)) planes[d] = S.size(d);
	}
}

// glut light sourse
void glut_light() {
	stim::vec3<float> p1 = cam.getLookAt() + cam.getUp() * 100000;
	stim::vec3<float> p2 = cam.getPosition();

	// light source
	GLfloat global_ambient[] = { 0.4, 0.4, 0.4, 1.0 };
	GLfloat ambient[] = { 0.2, 0.2, 0.2, 1.0 };
	GLfloat diffuse1[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat diffuse2[] = { 0.4, 0.4, 0.4, 1.0 };
	GLfloat specular[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat position1[] = { p1[0], p1[1], p1[2], 1.0 };		// upper right light source
	GLfloat position2[] = { p2[0], p2[1], p2[2], 1.0 };		// lower left light source

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glShadeModel(GL_SMOOTH);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);

	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);				// set ambient for light 0
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse1);				// set diffuse for light 0
	glLightfv(GL_LIGHT0, GL_SPECULAR, specular);			// set specular for light 0
	glLightfv(GL_LIGHT0, GL_POSITION, position1);			// set position for light 0

	glLightfv(GL_LIGHT1, GL_AMBIENT, ambient);				// set ambient for light 1
	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse2);				// set diffuse for light 1
	glLightfv(GL_LIGHT1, GL_SPECULAR, specular);			// set specular for light 1
	glLightfv(GL_LIGHT1, GL_POSITION, position2);			// set position for light 1
}

// dynamically set up menu
void glut_set_menu(int value) {

	// remove last time menu options then add new menu options
	switch (value) {
	case 1:
		for (int i = 0; i < 4; i++)
			glutRemoveMenuItem(1);
		if (open_light) {
			menu_name = main_menu_option[1];
			glutAddMenuEntry(menu_name.c_str(), 2);
			menu_name = sub_menu_option[5];
			glutAddMenuEntry(menu_name.c_str(), 5);
		}
		else {
			menu_name = sub_menu_option[0];
			glutAddMenuEntry(menu_name.c_str(), 6);
			for (int i = 1; i < 3; i++) {
				menu_name = main_menu_option[i];
				glutAddMenuEntry(menu_name.c_str(), i + 1);
			}
			menu_name = sub_menu_option[4];
			glutAddMenuEntry(menu_name.c_str(), 4);
		}
		break;
	case 2:
		for (int i = 0; i < 4; i++)
			glutRemoveMenuItem(1);
		if (open_light) {
			menu_name = main_menu_option[0];
			glutAddMenuEntry(menu_name.c_str(), 1);
			menu_name = sub_menu_option[2];
			glutAddMenuEntry(menu_name.c_str(), 8);
			menu_name = sub_menu_option[5];
			glutAddMenuEntry(menu_name.c_str(), 5);
		}
		else {
			menu_name = main_menu_option[0];
			glutAddMenuEntry(menu_name.c_str(), 1);
			menu_name = sub_menu_option[2];
			glutAddMenuEntry(menu_name.c_str(), 8);
			menu_name = main_menu_option[2];
			glutAddMenuEntry(menu_name.c_str(), 3);
			menu_name = sub_menu_option[4];
			glutAddMenuEntry(menu_name.c_str(), 4);
		}
		break;
	case 3:
		for (int i = 0; i < 4; i++)
			glutRemoveMenuItem(1);
		menu_name = main_menu_option[0];
		glutAddMenuEntry(menu_name.c_str(), 1);
		menu_name = main_menu_option[1];
		glutAddMenuEntry(menu_name.c_str(), 2);
		break;
	case 4:
		for (int i = 0; i < 4; i++)
			glutRemoveMenuItem(1);
		if (mapping_mode) {
			if (highlight_difference) {
				menu_name = sub_menu_option[3];
				glutAddMenuEntry(menu_name.c_str(), 9);
			}
			else {
				menu_name = main_menu_option[0];
				glutAddMenuEntry(menu_name.c_str(), 1);
				menu_name = sub_menu_option[2];
				glutAddMenuEntry(menu_name.c_str(), 8);
			}
		}
		else {
			menu_name = main_menu_option[1];
			glutAddMenuEntry(menu_name.c_str(), 2);
		}
		menu_name = sub_menu_option[5];
		glutAddMenuEntry(menu_name.c_str(), 5);
		break;
	case 5:
		for (int i = 0; i < 4; i++)
			glutRemoveMenuItem(1);
		if (compare_mode) {
			menu_name = sub_menu_option[0];
			glutAddMenuEntry(menu_name.c_str(), 6);
			for (int i = 1; i < 3; i++) {
				menu_name = main_menu_option[i];
				glutAddMenuEntry(menu_name.c_str(), i + 1);
			}
		}
		if (mapping_mode) {
			if (highlight_difference) {
				menu_name = sub_menu_option[3];
				glutAddMenuEntry(menu_name.c_str(), 9);
			}
			else {
				menu_name = main_menu_option[0];
				glutAddMenuEntry(menu_name.c_str(), 1);
				menu_name = sub_menu_option[2];
				glutAddMenuEntry(menu_name.c_str(), 8);
				menu_name = main_menu_option[2];
				glutAddMenuEntry(menu_name.c_str(), 3);
			}
		}
		menu_name = sub_menu_option[4];
		glutAddMenuEntry(menu_name.c_str(), 4);
		break;
	case 6:
		for (int i = 0; i < 3; i++)
			glutRemoveMenuItem(2);
		menu_name = sub_menu_option[1];
		glutChangeToMenuEntry(1, menu_name.c_str(), 7);
		break;
	case 7:
		menu_name = sub_menu_option[0];
		glutChangeToMenuEntry(1, menu_name.c_str(), 6);
		for (int i = 1; i < 3; i++) {
			menu_name = main_menu_option[i];
			glutAddMenuEntry(menu_name.c_str(), i + 1);
		}
		if (open_light) {
			menu_name = sub_menu_option[5];
			glutAddMenuEntry(menu_name.c_str(), 5);
		}
		else {
			menu_name = sub_menu_option[4];
			glutAddMenuEntry(menu_name.c_str(), 4);
		}
		break;
	case 8:
		for (int i = 0; i < 4; i++)
			glutRemoveMenuItem(1);
		if (open_light) {
			menu_name = sub_menu_option[3];
			glutAddMenuEntry(menu_name.c_str(), 9);
			menu_name = sub_menu_option[5];
			glutAddMenuEntry(menu_name.c_str(), 5);
		}
		else {
			menu_name = sub_menu_option[3];
			glutAddMenuEntry(menu_name.c_str(), 9);
			menu_name = sub_menu_option[4];
			glutAddMenuEntry(menu_name.c_str(), 4);
		}
		break;
	case 9:
		for (int i = 0; i < 4; i++)
			glutRemoveMenuItem(1);
		if (open_light) {
			menu_name = main_menu_option[0];
			glutAddMenuEntry(menu_name.c_str(), 1);
			menu_name = sub_menu_option[2];
			glutAddMenuEntry(menu_name.c_str(), 8);
			menu_name = sub_menu_option[5];
			glutAddMenuEntry(menu_name.c_str(), 5);
		}
		else {
			menu_name = main_menu_option[0];
			glutAddMenuEntry(menu_name.c_str(), 1);
			menu_name = sub_menu_option[2];
			glutAddMenuEntry(menu_name.c_str(), 8);
			menu_name = main_menu_option[2];
			glutAddMenuEntry(menu_name.c_str(), 3);
			menu_name = sub_menu_option[4];
			glutAddMenuEntry(menu_name.c_str(), 4);
		}
		break;
	}
}

// defines camera motion based on mouse dragging
void glut_motion(int x, int y) {

	int mods = glutGetModifiers();
	if (LButtonDown == true && RButtonDown == false && mods == 0) {

		float theta = orbit_factor * (mouse_x - x);		// determine the number of degrees along the x-axis to rotate
		float phi = orbit_factor * (y - mouse_y);		// number of degrees along the y-axis to rotate

		cam.OrbitFocus(theta, phi);						// rotate the camera around the focal point
	}
	else if (mods != 0) {
		float dx = (float)(x - mouse_x);
		float dist = dx;								// calculate the distance that the mouse moved in pixel coordinates
		float sdist = dist;								// scale the distance by the sensitivity
		if (mods == GLUT_ACTIVE_SHIFT) {				// if the SHIFT key is pressed
			planes[0] += (sdist)* S.spacing(0);			// move the X plane based on the mouse wheel direction
		}
		else if (mods == GLUT_ACTIVE_CTRL) {			// if the CTRL key is pressed
			planes[1] += (sdist)* S.spacing(1);			// move the Y plane based on the mouse wheel direction
		}
		else if (mods == GLUT_ACTIVE_ALT) {				// if hte ALT key is pressed
			planes[2] += (sdist)* S.spacing(2);			// move the Z plane based on the mouse wheel direction
		}
		enforce_bounds();
	}

	mouse_x = x;										// update the mouse position
	mouse_y = y;

	glutPostRedisplay();								// re-draw the visualization
}

// sets the menu options
void glut_menu(int value) {

	if (value == 1) {									// menu 1 represents comparing mode
		compare_mode = true;
		mapping_mode = false;
		volume_mode = false;
	}
	if (value == 2) {									// menu 2 represents mapping mode
		compare_mode = false;
		mapping_mode = true;
		volume_mode = false;
	}
	if (value == 3) {									// menu 3 represents volume mode
		compare_mode = false;
		mapping_mode = false;
		volume_mode = true;
	}
	if (value == 4) {									// menu 4 represents open light
		open_light = true;
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_LIGHT1);
	}
	if (value == 5) {									// menu 5 represents close light
		open_light = false;
		glDisable(GL_LIGHTING);
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHT1);
	}
	if (value == 6)										// menu 6 represents render overlaid network
		render_overlaid_network = true;
	if (value == 7)										// menu 7 represents erase overlaid network
		render_overlaid_network = false;
	if (value == 8)										// menu 8 represents turn on highlight
		highlight_difference = true;
	if (value == 9)										// menu 9 represents turn off highlight
		highlight_difference = false;

	glut_set_menu(value);

	glutPostRedisplay();
}

// get click window coordinates
void glut_mouse(int button, int state, int x, int y) {

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		mouse_x = x;
		mouse_y = y;
		LButtonDown = true;
	}
	else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		mouse_x = x;
		mouse_y = y;
		RButtonDown = true;
	}
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
		mouse_x = x;
		mouse_y = y;
		LButtonDown = false;
	}
	else if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) {
		mouse_x = x;
		mouse_y = y;
		RButtonDown = false;
	}
}

// define camera move based on mouse wheel move
void glut_wheel(int wheel, int direction, int x, int y) {

	int mods = glutGetModifiers();
	if (mods == GLUT_ACTIVE_SHIFT) {					// if the SHIFT key is pressed
		planes[0] += (direction)* S.spacing(0);			// move the X plane based on the mouse wheel direction
	}
	else if (mods == GLUT_ACTIVE_CTRL) {				// if the CTRL key is pressed
		planes[1] += (direction)* S.spacing(1);			// move the Y plane based on the mouse wheel direction
	}
	else if (mods == GLUT_ACTIVE_ALT) {					// if hte ALT key is pressed
		planes[2] += (direction)* S.spacing(2);			// move the Z plane based on the mouse wheel direction
	}
	else {
		if (direction > 0)								// if it is button 3(up), move closer
			delta = zoom_factor;
		else											// if it is button 4(down), leave farther
			delta = -zoom_factor;
	}
	cam.Push(delta);
	enforce_bounds();
	glutPostRedisplay();
}

// define keyboard inputs
void glut_keyboard(unsigned char key, int x, int y) {

	// register different keyboard operation
	switch (key) {

	// zooming
	case 'w':						// if keyboard 'w' is pressed, then move closer
		delta = zoom_factor;
		cam.Push(delta);
		break;
	case 's':						// if keyboard 's' is pressed, then leave farther
		delta = -zoom_factor;
		cam.Push(delta);
		break;

	// resample and re-render the cylinder in different radius
	case 'd':						// if keyboard 'd' is pressed, then increase radius by radius_factor
		radius += radius_factor;
		break;
	case 'a':						// if keyboard 'a' is pressed, then decrease radius by radius_factor
		radius -= radius_factor;
		// get rid of the degenerated case when radius decrease below 0
		if (radius < 0.001f)
			radius = 0.2;
		break;

	// close window and exit application
	case 27:						// if keyboard 'ESC' is pressed, then exit
		exit(0);
	}
	glutPostRedisplay();
}

// main render function
void glut_render() {
	
	glut_light();												// set up light
	
	if (num_nets == 1) {										// if a single network is loaded
		glEnable(GL_DEPTH_TEST);								// enable depth
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		// clear the screen
		glut_render_single_projection();						// fill the entire viewport
		glut_render_modelview();								// set up the modelview matrix with camera details
		if (volume_mode) {
			draw_box();
			draw_frames();
			glEnable(GL_TEXTURE_3D);							// enable 3D texture mapping
			S.bind();											// bind the texture
			draw_x_slice(planes[0]);							// draw the X plane
			draw_y_slice(planes[1]);							// draw the Y plane
			draw_z_slice(planes[2]);							// draw the Z plane
			glDisable(GL_TEXTURE_3D);							// disable 3D texture mapping
		}
		GT.glCenterline0();										// render the GT network (the only one loaded)
		glDisable(GL_DEPTH_TEST);
	}

	else if (num_nets == 2) {									// if two networks are loaded
		glEnable(GL_DEPTH_TEST);								// enable depth
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		// clear the screen
		
		// left projection
		glut_render_left_projection();							// set up a projection for the left half of the window
		glut_render_modelview();								// set up the modelview matrix using camera details
		if (compare_mode) {										// compare mode
			glEnable(GL_TEXTURE_1D);							// enable texture mapping
			if (!open_light)
				glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);	// texture map will be used as the network color
			else
				glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);// map light to texture

			_GT.glCylinder(sigma, radius);						// render the GT network
			if (render_overlaid_network) {
				glDisable(GL_TEXTURE_1D);							// temporarily disable texture
				glEnable(GL_BLEND);									// enable color blend
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	// set blend function
				glDisable(GL_DEPTH_TEST);							// should disable depth
				glColor4f(0.8f, 0.8f, 0.8f, 0.2f);
				_T.glAdjointCylinder(sigma, radius);
				glDisable(GL_BLEND);
				glEnable(GL_DEPTH_TEST);
				glEnable(GL_TEXTURE_1D);							// re-enable texture
				glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
			}
		}
		else if (mapping_mode) {
			glEnable(GL_COLOR_MATERIAL);
			if (!highlight_difference)
				_GT.glRandColorCylinder(0, _gt_t, colormap, sigma, radius);
			else
				_GT.glDifferenceCylinder(0, _gt_t, colormap, sigma, radius);
		}
		else if (volume_mode) {
			draw_box();
			draw_frames();
			glEnable(GL_TEXTURE_3D);								// enable 3D texture mapping
			S.bind();												// bind the texture
			draw_x_slice(planes[0]);								// draw the X plane
			draw_y_slice(planes[1]);								// draw the Y plane
			draw_z_slice(planes[2]);								// draw the Z plane
			glDisable(GL_TEXTURE_3D);								// disable 3D texture mapping
			_GT.glCylinder(sigma, radius);
		}

		// right projection
		glut_render_right_projection();							// set up a projection for the right half of the window
		glut_render_modelview();								// set up the modelview matrix using camera details
		if (compare_mode) {										// compare mode
			_T.glCylinder(sigma, radius);						// render the GT network
			glDisable(GL_TEXTURE_1D);
			if (render_overlaid_network) {
				glEnable(GL_BLEND);									// enable color blend
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	// set blend function
				glDisable(GL_DEPTH_TEST);							// should disable depth
				glColor4f(0.8f, 0.8f, 0.8f, 0.2f);
				_GT.glAdjointCylinder(sigma, radius);
				glDisable(GL_BLEND);
				glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
			}
		}
		else if (mapping_mode) {
			if (!highlight_difference)
				_T.glRandColorCylinder(1, _t_gt, colormap, sigma, radius);
			else
				_T.glDifferenceCylinder(1, _t_gt, colormap, sigma, radius);
			glDisable(GL_COLOR_MATERIAL);
		}
		else if (volume_mode) {
			draw_box();
			draw_frames();
			glEnable(GL_TEXTURE_3D);								// enable 3D texture mapping
			S.bind();												// bind the texture
			draw_x_slice(planes[0]);								// draw the X plane
			draw_y_slice(planes[1]);								// draw the Y plane
			draw_z_slice(planes[2]);								// draw the Z plane
			glDisable(GL_TEXTURE_3D);								// disable 3D texture mapping
			_T.glCylinder(sigma, radius);
		}
		sigma = radius;
	}

	if (num_nets == 2) {												// works only with two networks
		std::ostringstream ss;
		if (mapping_mode)												// if it is in mapping mode
			ss << "Mapping Mode";
		else if (compare_mode)
			ss << "Compare Mode";										// default mode is compare mode
		else
			ss << "Volume Display";

		if (open_light)
			glDisable(GL_LIGHTING);
		glMatrixMode(GL_PROJECTION);									// set up the 2d viewport for mode text printing
		glPushMatrix();
		glLoadIdentity();
		int X = glutGet(GLUT_WINDOW_WIDTH);								// get the current window width
		int Y = glutGet(GLUT_WINDOW_HEIGHT);							// get the current window height
		glViewport(0, 0, X / 2, Y);										// locate to left bottom corner
		gluOrtho2D(0, X, 0, Y);											// define othogonal aspect
		glColor3f(0.8f, 0.0f, 0.0f);									// using red to show mode

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();

		glRasterPos2f(0, 5);											//print text in the left bottom corner
		glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, (const unsigned char*)(ss.str().c_str()));

		glPopMatrix();
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glColor3f(1.0f, 1.0f, 1.0f);									//clear red color
		if (open_light)
			glEnable(GL_LIGHTING);
	}

	glutSwapBuffers();
}

#define BREWER_CTRL_PTS 11										// number of control points in the Brewer map
void texture_initialize() {

	//define the colormap
	static float  brewer_map[BREWER_CTRL_PTS][3] = {			// generate a Brewer color map (blue to red)
		{ 0.192157f, 0.211765f, 0.584314f },
		{ 0.270588f, 0.458824f, 0.705882f },
		{ 0.454902f, 0.678431f, 0.819608f },
		{ 0.670588f, 0.85098f, 0.913725f },
		{ 0.878431f, 0.952941f, 0.972549f },
		{ 1.0f, 1.0f, 0.74902f },
		{ 0.996078f, 0.878431f, 0.564706f },
		{ 0.992157f, 0.682353f, 0.380392f },
		{ 0.956863f, 0.427451f, 0.262745f },
		{ 0.843137f, 0.188235f, 0.152941f },
		{ 0.647059f, 0.0f, 0.14902f }
	};

	glGenTextures(1, &cmap_tex);								// generate a texture map name
	glBindTexture(GL_TEXTURE_1D, cmap_tex);						// bind the texture map

	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);		// enable linear interpolation
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);			// clamp the values at the minimum and maximum
	glTexImage1D(GL_TEXTURE_1D, 0, 3, BREWER_CTRL_PTS, 0, GL_RGB, GL_FLOAT,	// upload the texture map to the GPU
		brewer_map);
	if (load_image_stack == 1) {
		S.attach();												// attach 3D texture
	}
}

// initialize the OpenGL (GLUT) window, including starting resolution, callbacks, texture maps, and camera
void glut_initialize() {

	int myargc = 1;												// GLUT requires arguments, so create some bogus ones
	char* myargv[1];
	myargv[0] = strdup("netmets");

	glutInit(&myargc, myargv);									// pass bogus arguments to glutInit()
	glutSetOption(GLUT_MULTISAMPLE, 8);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);	// generate a color buffer, depth buffer, and enable double buffering
	glutInitWindowPosition(100, 100);							// set the initial window position
	glutInitWindowSize(320, 320);								// set the initial window size
	glutCreateWindow("NetMets - STIM Lab, UH");					// set the dialog box title

#ifdef _WIN32
	GLenum err = glewInit();									// initialize GLEW (necessary for Windows)
	if (GLEW_OK != err) {										// eror with GLEW
		std::cout << "Error with GLEW: " << glewGetErrorString(err) << std::endl;
		exit(1);
	}
#endif

	// register callback functions
	glutDisplayFunc(glut_render);				// function executed for rendering - renders networks
	glutMouseFunc(glut_mouse);					// executed on a mouse click - sets starting mouse positions for rotations
	glutMotionFunc(glut_motion);				// executed when the mouse is moved while a button is pressed
	glutKeyboardFunc(glut_keyboard);			// register keyboard callback
	glutMouseWheelFunc(glut_wheel);				// register mouse wheel callback

	// set up initial menu
	glutCreateMenu(glut_menu);					// register menu option callback
	for (int i = 0; i < 3; i++) {
		menu_name = main_menu_option[i];
		glutAddMenuEntry(menu_name.c_str(), i + 1);
	}
	menu_name = sub_menu_option[4];
	glutAddMenuEntry(menu_name.c_str(), 5);
	menu_name = sub_menu_option[0];
	glutChangeToMenuEntry(1, menu_name.c_str(), 6);
	glutAttachMenu(GLUT_RIGHT_BUTTON);			// register right mouse to open menu option

	texture_initialize();						// set up texture mapping (create texture maps, enable features)

	stim::vec3<float> c = bb.center();			// get the center of the network bounding box
	// place the camera along the z-axis at a distance determined by the network size along x and y
	cam.setPosition(c + stim::vec<float>(0, 0, camera_factor * std::max(bb.size()[0], bb.size()[1])));
	cam.LookAt(c[0], c[1], c[2]);				// look at the center of the network
}


//********************segmentation fucntion********************
// set up device on gpu
#ifdef __CUDACC__
// set specific device to work on
void set_device(int &device) {
	int count;
	cudaGetDeviceCount(&count);					// numbers of device that are available
	if (count < device + 1) {
		std::cout << "No such device available, please set another device" << std::endl;
		exit(1);
	}
}
#else
void set_device(int &device) {
	device = -1;								// set to default -1
}
#endif

// split and map two networks and fill the networks' R with metric information
void mapping(float sigma, int device, float threshold) {

	//GT = GT.compare(T, sigma, device);				// compare the ground truth to the test case - store errors in GT
	//T = T.compare(GT, sigma, device);				// compare the test case to the ground truth - store errors in T

	// compare and split two networks
	_GT.split(GT, T, sigma, device, threshold);
	_T.split(T, GT, sigma, device, threshold);

	// mapping two new splitted networks and get their edge relation
	_GT.mapping(_T, _gt_t, device, threshold);
	_T.mapping(_GT, _t_gt, device, threshold);

	// generate random color set based on the number of edges in GT
	size_t num = _gt_t.size();						// also create random color for unmapping edge, but won't be used though
	colormap.resize(3 * num);						// 3 portions compound RGB
	for (int i = 0; i < 3 * num; i++)
		colormap[i] = rand() / (float)RAND_MAX;		// set to [0, 1]

	float FPR = _GT.average();						// calculate the average metric
	float FNR = _T.average();

	std::cout << "FNR: " << FPR << std::endl;		// print false alarms and misses
	std::cout << "FPR: " << FNR << std::endl;
}

// writes features of the networks i.e average segment length, tortuosity, branching index, contraction, fractal dimension, number of end and branch points to a csv file
void features(std::string filename) {
	double avgL_t, avgL_gt, avgT_t, avgT_gt, avgB_t, avgB_gt, avgC_t, avgC_gt, avgFD_t, avgFD_gt;
	unsigned int e_t, e_gt, b_gt, b_t;
	avgL_gt = GT.Lengths();
	avgT_gt = GT.Tortuosities();
	avgL_t = T.Lengths();
	avgT_t = T.Tortuosities();
	avgB_gt = GT.BranchingIndex();
	avgB_t = T.BranchingIndex();
	avgC_gt = GT.Contractions();
	avgFD_gt = GT.FractalDimensions();
	avgC_t = T.Contractions();
	avgFD_t = T.FractalDimensions();
	e_gt = GT.EndP();
	e_t = T.EndP();
	b_gt = GT.BranchP();
	b_t = T.BranchP();
	std::ofstream myfile;
	myfile.open(filename.c_str());
	myfile << "Length, Tortuosity, Contraction, Fractal Dimension, Branch Points, End points, Branching Index, \n";
	myfile << avgL_gt << "," << avgT_gt << "," << avgC_gt << "," << avgFD_gt << "," << b_gt << "," << e_gt << "," << avgB_gt << std::endl;
	myfile << avgL_t << "," << avgT_t << "," << avgC_t << "," << avgFD_t << "," << b_t << "," << e_t << "," << avgB_t << std::endl;
	myfile.close();
}

// output an advertisement for the lab, authors, and usage information
void advertise() {
	std::cout << std::endl << std::endl;
	std::cout << "=========================================================================" << std::endl;
	std::cout << "Thank you for using the NetMets network comparison tool!" << std::endl;
	std::cout << "Scalable Tissue Imaging and Modeling (STIM) Lab, University of Houston" << std::endl;
	std::cout << "Developers: Jiaming Guo, David Mayerich" << std::endl;
	std::cout << "Source: https://git.stim.ee.uh.edu/segmentation/netmets" << std::endl;
	std::cout << "=========================================================================" << std::endl << std::endl;

	std::cout << "usage: netmets file1 file2 --sigma 3" << std::endl;
	std::cout << "            compare two .obj files with a tolerance of 3 (units defined by the network)" << std::endl << std::endl;
	std::cout << "       netmets file1 --gui" << std::endl;
	std::cout << "            load a file and display it using OpenGL" << std::endl << std::endl;
	std::cout << "       netmets file1 file2 --device 0" << std::endl;
	std::cout << "            compare two files using device 0 (if there isn't a gpu, use cpu)" << std::endl << std::endl;
	std::cout << "       netmets file1 file2 --mapping value" << std::endl;
	std::cout << "            mapping two files in random colors with a threshold of value" << std::endl << std::endl;
}

int main(int argc, char* argv[])
{
	stim::arglist args;						// create an instance of arglist

	// add arguments
	args.add("help", "prints this help");
	args.add("sigma", "force a sigma value to specify the tolerance of the network comparison", "3");
	args.add("gui", "display the network or network comparison using OpenGL");
	args.add("device", "choose specific device to run", "0");
	args.add("features", "save features to a CSV file, specify file name");
	args.add("threshold", "metric acceptable value", "0.6", "any real positive value");
	args.add("stack", "load the image stacks");
	args.add("spacing", "spacing between pixel samples in each dimension", "1.0 1.0 1.0", "any real positive value");

	args.parse(argc, argv);					// parse the user arguments

	if (args["help"].is_set()) {			// test for help
		advertise();						// output the advertisement
		std::cout << args.str();			// output arguments
		exit(1);							// exit
	}

	if (args.nargs() >= 1) {				// if at least one network file is specified
		num_nets = 1;						// set the number of networks to one
		std::vector<std::string> tmp = stim::parser::split(args.arg(0), '.');	// split the filename at '.'
		if ("swc" == tmp[1]) 				// loading swc file
			GT.load_swc(args.arg(0));		// load the specified file as the ground truth
		else if ("obj" == tmp[1])			// loading obj file
			GT.load_obj(args.arg(0));		// load the specified file as the ground truth
		else if ("nwt" == tmp[1])			// loading nwt file
			GT.loadNwt(args.arg(0));
		else {
			std::cout << "Invalid loading file" << std::endl;
			exit(1);
		}
	}

	if (args.nargs() == 2) {					// if two files are specified, they will be displayed in neighboring viewports and compared
		num_nets = 2;							// set the number of networks to two

		int device = args["device"].as_int();	// get the device value from the user
		set_device(device);

		sigma = args["sigma"].as_float();		// get the sigma value from the user

		if (args["features"].is_set())			// if the user wants to save features
			features(args["features"].as_string());

		threshold = args["threshold"].as_float();

		std::vector<std::string> tmp = stim::parser::split(args.arg(1), '.');	// split the filename at '.'
		if ("swc" == tmp[1]) 					// loading swc files
			T.load_swc(args.arg(1));            // load the second (test) network
		else if ("obj" == tmp[1])				// loading obj files
			T.load_obj(args.arg(1));
		else if ("nwt" == tmp[1])				// loading nwt file
			T.loadNwt(args.arg(1));
		else {
			std::cout << "Invalid loading file" << std::endl;
			exit(1);
		}

		GT = GT.resample(resample_rate * sigma);// resample both networks based on the sigma value
		T = T.resample(resample_rate * sigma);

		mapping(sigma, device, threshold);
	}

	// load image stack
	if (args["stack"].is_set()) {
		S.load_images(args["stack"].as_string());
		load_image_stack = true;
	}

	// set up spacing value, the vexel
	float sp[3] = { 1.0f, 1.0f, 1.0f };						// allocate variables for grid spacing
	if (args["spacing"].nargs() == 1)						// if only one argument is given
		sp[2] = (float)args["spacing"].as_float(0);			// assume that it's the z coordinate (most often anisotropic)
	else if (args["spacing"].nargs() == 3) {				// if three arguments are given
		sp[0] = (float)args["spacing"].as_float(0);			// set the arguments as expected
		sp[1] = (float)args["spacing"].as_float(1);
		sp[2] = (float)args["spacing"].as_float(2);
	}
	S.spacing(sp[0], sp[1], sp[2]);							// set the spacing between samples

	// set start plane at one quater
	planes[0] = S.size(0) / 4.0f;							// initialize the start positions for the orthogonal display planes
	planes[1] = S.size(1) / 4.0f;
	planes[2] = S.size(2) / 4.0f;

	//if a GUI is requested, display the network using OpenGL
	if (args["gui"].is_set()) {

		bb = GT.boundingbox();					// generate a bounding volume		
		glut_initialize();						// create the GLUT window and set callback functions		
		glutMainLoop();							// enter GLUT event processing cycle
	}
}