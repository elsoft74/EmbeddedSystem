#define R1 2
#define G1 4
#define Y1 7
#define BU 9

bool SR1;
bool SG1;
bool SY1;
int SBU;
int setting[]={0x02,0x03,0x04,0x05};
String frequencies[]={"3921.16","490.20","122.55","30.64"};
int conf;
int incomingByte = 0;
int wd=10;

void setup() {
  Serial.begin(9600);
  SR1 = false;
  SG1 = false;
  SY1 = false;
  SBU=0;
  conf=1;
  setLedStatus(SR1,SG1, SY1);
  TCCR1B = TCCR1B & B11111000 | setting[conf];
  analogWrite(BU,SBU);
}

void loop() {
  SR1 = random(2);
  SG1 = random(2);
  SY1 = random(2);

  printStatus();
  setLedStatus(SR1,SG1, SY1);
  setBuzzer(SR1,SG1, SY1);
  if (Serial.available() > 0) {
    incomingByte = Serial.read();
    Serial.print("Received:");
    Serial.print(incomingByte);
    Serial.println();
    if(incomingByte>48 && incomingByte<53){
      conf=incomingByte-48;
      TCCR1B = TCCR1B & B11111000 | setting[conf];
    }
    if(incomingByte==45 && wd>10){
      wd-=5;
    }
    if(incomingByte==43 && wd<1000){
      wd+=5;
    }
  }
}

void setLedStatus(bool r,bool g, bool y){
  digitalWrite(R1,r);
  digitalWrite(G1,g);
  digitalWrite(Y1,y);
}
  
void setBuzzer(bool r,bool g, bool y){
  //SBU = random(40);
  if(!(r||g||y)){
    SBU=0;
  } else {
    SBU = 42;
    }
  if(y){
    SBU*=2;
    }
  if (r){
    SBU*=3;
    }
  SBU=1+SBU%255;
  for (int i=0;i<SBU;i++){
    analogWrite(BU,i);
    delay(wd);
    }
  for (int i=SBU;i>0;i--){
    analogWrite(BU,i);
    delay(wd);
  }
    //delay(DT);  
}
  
void printStatus(){
  Serial.print("R1:");
  Serial.print(SR1);
  Serial.print("\t");
  Serial.print("G1:");
  Serial.print(SG1);
  Serial.print("\t");
  Serial.print("Y1:");
  Serial.print(SY1);
  Serial.print("\t");
  Serial.print("BU:");
  Serial.print(SBU);
  Serial.print("\t");
  Serial.print("FRQ:");
  Serial.print(frequencies[conf]);
  Serial.print(" Hz");
  Serial.print("\t");
  Serial.print("Fading waiting delay");
  Serial.print(wd);
  Serial.print(" ms");
  Serial.println();
}
