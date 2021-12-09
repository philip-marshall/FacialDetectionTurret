#include <Servo.h>

int xcoordinate = 0;
int ycoordinate = 0;
String serialInfo;

Servo xServo;
Servo yServo;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.setTimeout(100);

  xServo.attach(A0);
  yServo.attach(A1);

  xServo.write(90);
  yServo.write(90);
}

void loop() 
{
}

void serialEvent() {
  if (Serial.available())
  {
    serialInfo = Serial.readString();
    
    xcoordinate = parseX(serialInfo);
    ycoordinate = parseY(serialInfo);
    Serial.println(xcoordinate);
    xServo.write(xcoordinate);
    yServo.write(ycoordinate);
  }
}

int parseX(String coords){
  coords.remove(coords.indexOf("Y"));
  coords.remove(coords.indexOf("X", 1));

  return coords.toInt();
}

int parseY(String coords)
{
  coords.remove(0, coords.indexOf("Y") + 1);
  return coords.toInt();
}
