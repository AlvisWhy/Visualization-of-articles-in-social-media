class roadLine {

  PVector startP, endP;

  roadLine(PVector s, PVector e) {
    startP = s;
    endP = e;
  }

  void show() {
    pushStyle();
    strokeWeight(0.25);
    stroke(255,0,0, 50);
    line(startP.x, startP.y, startP.z, endP.x, endP.y, endP.z);
    popStyle();
  }
}
