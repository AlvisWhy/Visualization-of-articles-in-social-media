class way {

  PVector startP, endP;
  float seglength = 30;
  float crossWidth;
  float hei;
  int commentNum;
  float r, g, b;
  ArrayList<String> articles;
  ArrayList<comment> comments;


  way(PVector s, PVector e) {
    startP = s;
    endP = e;
    comments = new ArrayList<comment>();
    commentNum = int(random(5, 10));

    r = random(100, 255);
    g = random(0, 155);
    b = random(100, 205);
  }

  void attract() {
  }

  void show() {
    //rect(1,1,1,1,1,1);
    pushStyle();

    color k = color(r, g, b, 50);

    fill(k);
    strokeWeight(0.25);
    stroke(255, 100);
    line(startP.x, startP.y, startP.z, endP.x, endP.y, endP.z);
    noStroke();

    beginShape();

    vertex(startP.x, startP.y, startP.z);

    vertex(endP.x, endP.y, endP.z);

    vertex(endP.x, endP.y, endP.z + seglength);

    vertex(startP.x, startP.y, startP.z +seglength);

    endShape();
    popStyle();
  }


  void lShow() {
    //rect(1,1,1,1,1,1);
    strokeWeight(0.4);
    stroke(255, 50);


    strokeWeight(1);

    stroke(0);
    line(startP.x, startP.y, startP.z, startP.x, startP.y, startP.z + 50);
    line(endP.x, endP.y, endP.z, endP.x, endP.y, endP.z + 50);
    pushStyle();

 
   fill(255, 255, 255, 10);
    stroke(255, 100);
    //line(startP.x, startP.y, startP.z, endP.x, endP.y, endP.z);
    noStroke();

    beginShape();

    vertex(startP.x, startP.y, startP.z);

    vertex(endP.x, endP.y, endP.z);

    vertex(endP.x, endP.y, endP.z + seglength);

    vertex(startP.x, startP.y, startP.z +seglength);

    endShape();
    popStyle();
  }

  void textrues() {
  }
}
