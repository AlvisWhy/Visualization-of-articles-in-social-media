class article {

  boolean ifRelation;
  PVector location;
  int index;
  int relationIndex;
  int relationType;
  float sizeX, sizeY;
  String name;
  float type;
  PImage p;
  PImage icon;
  int attachMode;
  int nameLine;
  way attach;
  
  float fff;

  int infor_layer;

  int influence = 1;


  article(float a, float b, boolean c, String d, int k, int ind, int rInd ) {
    sizeX = a;
    sizeY = b;
    ifRelation =c;

    name = d;
    relationType = k;
    p = loadImage("w (" + String.valueOf(int(random(1, 140)))+ ").png");
    icon = loadImage("icon" +".png");
    index = ind;
    if (ifRelation == true) {
      relationIndex = rInd;
    }
    type = random(1);
    location = new PVector(random(0, 900), random(0, 900), 0);
  }

  void arrange(article resource, float dist, float ranDistX, float ranDistY) {
    if (ifRelation == true) {

      location.z = resource.location.z+ dist;
      location.x = resource.location.x+ ranDistX;
      location.y = resource.location.y+ ranDistY;

      if (relationType == 0) {
        resource.influence += 4;
      } else if (relationType == 1) {
        resource.influence += 2;
      } else if (relationType == 2) {
        resource.influence += 1;
      }
    } else {
      location.z = 0;
    }
  }

  void drawLine(article resource) {
    pushStyle();
    noFill();
    stroke(246, 107, 121, 150);
    if (relationType == 0) {
      strokeWeight(0.5);
    } else if (relationType == 1) {
      strokeWeight(1);
    } else if (relationType == 2) {
      strokeWeight(0.5);
    }

    if (type>0.5) {
      line(resource.location.x, resource.location.y, resource.location.z, resource.location.x, resource.location.y, location.z);
      line(location.x, location.y, location.z, resource.location.x, location.y, location.z);
      line(resource.location.x, location.y, location.z, resource.location.x, resource.location.y, location.z);
    } else {
      line(resource.location.x, resource.location.y, resource.location.z, resource.location.x, resource.location.y, location.z);
      line(location.x, location.y, location.z, location.x, resource.location.y, location.z);
      line(resource.location.x, resource.location.y, location.z, location.x, resource.location.y, location.z);
    }

    popStyle();
  }



  void attribute() {
    fff = random(10, 30);
    location.z += fff;
    if (attachMode == 0 || attachMode ==2) {
      location.x = attach.startP.x;
    } else {
      location.y = attach.startP.y;
    }
  }


  void show() {
    pushStyle();
    noFill();
    float SW = map(influence, 1, 20, 3, 15);
    strokeWeight(SW/2);
    noStroke();

    pushMatrix();
    translate(location.x, location.y, location.z);
    rotate(-PI/2);
    stroke(255,0,0, 200);
    strokeWeight(0.5);
    point(0, 0, 0);

    popMatrix();

    pushMatrix();
    translate(location.x, location.y, location.z);
    rotateX(-PI/2);
    if (attachMode == 1 || attachMode ==3) {
      rotateY(PI/2);
    }

    if (attachMode == 0) {
      translate(-20, 0, 0);
    } else  if (attachMode == 3) {
      translate(-20, 0,0 );
    } 
    
    else  if (attachMode == 3) {
      translate(0, -20, 0);
    } 
    fill(attach.r, attach.g, attach.b,100);
    strokeWeight(0.25);
    stroke(attach.r, attach.g, attach.b);
    rect(0, 0, 20, 10);


    scale(0.04);
    pushMatrix();
    translate(0,170,0);
    image(icon,0,0);
    popMatrix();
    image(p, 0, 0);
    popMatrix();

    popStyle();
  }



  void showText() {
    pushMatrix();
    pushStyle();
    noStroke();
    fill(255, 255, 255, 200);

    textSize(0.8);
    textAlign(LEFT);

    textLeading(1);
    translate(location.x, location.y, location.z);
    rotateX(-PI/2);



    if (attachMode == 1 || attachMode ==3) {
      rotateY(PI/2);
    }

    if (attachMode == 0) {
      translate(-15, 0, 0);
    } else  if (attachMode == 1) {
      translate(5, 0, 0);
    } else  if (attachMode == 2) {
      translate(5, 0, 0);
    } else  if (attachMode == 3) {
      translate(-15, 0, 0);
    } 


    text(name, 0, 0, 15, 10);

    popStyle();
    popMatrix();
  }
}
