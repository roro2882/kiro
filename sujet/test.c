#include <json/value.h>
#include <fstream>

std::ifstream people_file("tiny.json", std::ifstream::binary);
people_file >> data;

cout<<data; //This will print the entire json object.


