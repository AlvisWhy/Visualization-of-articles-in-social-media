class comment{
   PVector startP, endP;
  float leng;
  float Widt;
  float r,g,b;
  float seglength;
  
  String name;
  
  float widdd;
  int attachMode;
  comment(PVector a, PVector k, int l, String n){
    startP = a;
    endP = k;
    attachMode = l;
    name = n;
    r = random(0,255);
    g = random(0,255);
    b = random(0,255);
    seglength = random(2,5);
     widdd = random(4,7);
  }
  
  void show(){
    pushStyle();
    strokeWeight(0.25);
    stroke(255);
    point(startP.x,startP.y,startP.z);
    
    stroke(r,g,b,100);
    
    PVector dif = new PVector(endP.x - startP.x,endP.y- startP.y,0);
    dif.normalize();
    dif.mult(widdd);
    PVector dd = PVector.add(dif,startP);
    line(startP.x,startP.y,startP.z,endP.x,endP.y,startP.z);
    
    pushMatrix();
    noStroke();
    fill(r,g,b,50);
    
        if (attachMode == 0) {
      translate(-3, 0, 0);
    } else  if (attachMode == 1) {
      translate(3, 0, 0);
    } else  if (attachMode == 2) {
      translate(3, 0, 0);
    } else  if (attachMode == 3) {
      translate(-3, 0, 0);
    } 
    
    beginShape();
    
    vertex(startP.x, startP.y, startP.z);

    vertex(dd.x, dd.y,  startP.z);

    vertex(dd.x, dd.y, startP.z + seglength);

    vertex(startP.x, startP.y, startP.z +seglength);

    endShape();
    popMatrix();
    
     strokeWeight(2);
     stroke(r,g,b,50);
    point(startP.x,startP.y,startP.z);
    popStyle();
  }
  
   void showText() {
    pushMatrix();
    pushStyle();
    noStroke();
    fill(255, 255, 255, 200);
    textMode(MODEL);
    textSize(0.4);
    textAlign(CENTER);

    textLeading(1);
    translate(startP.x, startP.y, startP.z);
    rotateX(-PI/2);



    if (attachMode == 0 || attachMode ==2) {
      rotateY(PI/2);
    }

    if (attachMode == 0) {
      translate(0, 0, seglength);
    } else  if (attachMode == 1) {
      translate(0, 0, seglength);
    } else  if (attachMode == 2) {
      translate(0, 0, seglength);
    } else  if (attachMode == 3) {
      translate(0, 0, seglength);
    } 


    text(name, 0, 0, widdd, seglength);

    popStyle();
    popMatrix();
  }
}
