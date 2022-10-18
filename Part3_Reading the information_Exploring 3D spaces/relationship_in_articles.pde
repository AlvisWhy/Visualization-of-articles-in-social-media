
import peasy.*;
PeasyCam cam;
CameraState state;

ArrayList<article> articles = new ArrayList<article>();
ArrayList<way> base = new ArrayList<way>();
ArrayList<way> ways = new ArrayList<way>();
ArrayList<String> names = new ArrayList<String>();

ArrayList<roadLine> roadlines = new ArrayList<roadLine>();

int Mode = 0;
int AllI = 3000;
Table table;

int userNum;
float distance = 30;
float angle = 0;

float interval = 160;
float crosswidth = 15;
int col, row;

int counter = 0;
int count=1;
int roadlinenum = 6;
void setup() {

  size(1000, 1000, P3D);
  textMode(MODEL);

  cam = new PeasyCam(this, 100);
  state = cam.getState();

  col = int(height/ interval);
  row = int(width/interval);
  for (int k = 0; k< 12; k++) {
    for (int i =-1; i< row+1; i++) {
      for (int j =-1; j< col+1; j++) {
        float va = crosswidth/roadlinenum;
        for (int f = 1; f< roadlinenum; f++) {
          roadlines.add(new roadLine(new PVector(i*interval + va*f, j*interval, k*distance), new PVector(i*interval + va*f, (j+1)*interval, k*distance)));
          roadlines.add(new roadLine(new PVector((i+1)*interval - va*f, j*interval, k*distance), new PVector((i+1)*interval - va*f, (j+1)*interval, k*distance)));
          roadlines.add(new roadLine(new PVector(i*interval, j*interval+ va*f, k*distance), new PVector((i+1)*interval, j*interval+ va*f, k*distance)));
          roadlines.add(new roadLine(new PVector(i*interval, (j+1)*interval- va*f, k*distance), new PVector((i+1)*interval, (j+1)*interval- va*f, k*distance)));
        }

        base.add(new way(new PVector(i*interval + crosswidth, j*interval + crosswidth, k*distance), new PVector((i+1)*interval -crosswidth, j*interval+ crosswidth, k*distance)));
        base.add(new way(new PVector(i*interval+ crosswidth, j*interval+ crosswidth, k*distance), new PVector(i *interval+ crosswidth, (j+1)*interval- crosswidth, k*distance)));
        base.add(new way(new PVector((i+1)*interval - crosswidth, (j+1)*interval - crosswidth, k*distance), new PVector((i+1)*interval -crosswidth, j*interval+ crosswidth, k*distance)));
        base.add(new way(new PVector((i+1)*interval - crosswidth, (j+1)*interval - crosswidth, k*distance), new PVector(i *interval+ crosswidth, (j+1)*interval- crosswidth, k*distance)));
      }
    }
  }


  table = loadTable("t_comment.csv");
  for (TableRow row : table.rows()) {
    String k = row.getString(0);
    String m = "";
    char[] chars = k.toCharArray();
    int ct = 0;
    for (char ch : chars) {
      if (ch != '\n') {
        m+=ch;
      }
      if (ct%30 == 29) {
        m += '\n';
      }
      ct ++;
      if (ct>150) {
        break;
      }
    }
    names.add(m);
  }
}

void draw() {

  pushMatrix();
  translate(width/2, height*0.5, -100);
  scale(5);
  //rotateZ(PI/2);

  rotateX(PI*0.5);

  translate(-1*width/4, -25, -120);
  //scale(2);
  background(0);

  //setRandom
  //20%tran,30%interact,50%scan
  pushMatrix();
  translate(0, 0, 0);
  pushStyle();
  noStroke();
  fill(255, 50);
  translate(0, 0, 0);
  //rect(0, 0, 2900, 2900);

  translate(0, 0, distance);
  //rect(0, 0, 2900, 2900);
  popStyle();
  popMatrix();

  for (roadLine r : roadlines) {
    r.show();
  }

  for (way w : base) {
    w.lShow();
  }


  for (way w : ways) {
    w.show();


    for (comment k : w.comments) {
      k.show();
    }
  }

  int m;
  float ks= random(1);
  if (ks<0.2 || ks==0.2) {
    m = 0;
  } else if (ks>0.2 && ks<0.5) {
    m=1;
  } else {
    m =2;
  }

  //80%have relation
  boolean op = false;
  float kt = random(1);
  if (kt<0.8) {
    op = true;
  }

  //generate random source
  int numS = int(random( articles.size()));

  //setRandom
  if (counter <AllI) {
    articles.add(new article(8, 2, op, names.get( (articles.size())%(names.size()-1)), m, articles.size(), numS));

    // set the location

    articles.get(articles.size()-1).arrange(articles.get(numS), distance, random(-30, 30), random(-30, 30));

    article a =  articles.get(articles.size()-1);
    int i = int(a.location.x/interval);
    int j = int(a.location.y/interval);
    int numZ =int(a.location.z);

    int nearestEdge = 0;

    float dist0 = abs(a.location.x - i*interval);
    float dist1 = abs(a.location.y - j*interval);
    float dist2 = abs(a.location.x - (i+1)*interval);
    float dist3 = abs(a.location.y - (j+1)*interval);

    if (dist0 <= dist1 & dist0 <= dist2 & dist0<= dist3) {
      nearestEdge = 0;
    } else if (dist1 <= dist0 & dist1 <= dist2 & dist1<= dist3) {
      nearestEdge = 1;
    } else if (dist2 <= dist1 & dist2 <= dist0 & dist2<= dist3) {
      nearestEdge = 2;
    } else {
      nearestEdge = 3;
    }

    a.attachMode = nearestEdge;

    if (nearestEdge == 0) {
      way es = new way(new PVector(i*interval+ crosswidth, j*interval+ crosswidth, numZ), new PVector(i *interval+ crosswidth, (j+1)*interval- crosswidth, numZ));
      for (int t=0; t< es.commentNum; t++) {       
        if (random(1)>0.5) {
          es.comments.add(new comment(new PVector(random(es.startP.x, es.endP.x), random( es.startP.y, es.endP.y), es.startP.z+ random(30)), es.endP, nearestEdge, names.get( (articles.size()+2)%(names.size()-1))));
        } else {
          es.comments.add(new comment(new PVector(random(es.startP.x, es.endP.x), random( es.startP.y, es.endP.y), es.startP.z+ random(30)), es.startP, nearestEdge, names.get( (articles.size()+2)%(names.size()-1))));
        }
      }
      a.attach = es;

      if (! ways.contains(es)) {
        base.remove(es);
        ways.add(es);
      }
    } else if (nearestEdge == 1) {
      way es = new way(new PVector(i*interval + crosswidth, j*interval + crosswidth, numZ), new PVector((i+1)*interval -crosswidth, j*interval+ crosswidth, numZ));
      for (int t=0; t< es.commentNum; t++) {       
        if (random(1)>0.5) {
          es.comments.add(new comment(new PVector(random(es.startP.x, es.endP.x), random( es.startP.y, es.endP.y), es.startP.z+ random(30)), es.endP, nearestEdge, names.get( (articles.size()+2)%(names.size()-1))));
        } else {
          es.comments.add(new comment(new PVector(random(es.startP.x, es.endP.x), random( es.startP.y, es.endP.y), es.startP.z+ random(30)), es.startP, nearestEdge, names.get( (articles.size()+2)%(names.size()-1))));
        }
      }
      a.attach = es;

      if (! ways.contains(es)) {
        base.remove(es);
        ways.add(es);
      } else {
        print(1);
      }
    } else if (nearestEdge == 2) {
      way es = new way(new PVector((i+1)*interval - crosswidth, (j+1)*interval - crosswidth, numZ), new PVector((i+1)*interval -crosswidth, j*interval+ crosswidth, numZ));
      for (int t=0; t< es.commentNum; t++) {       
        if (random(1)>0.5) {
          es.comments.add(new comment(new PVector(random(es.endP.x, es.startP.x), random(es.endP.y, es.startP.y), es.startP.z+ random(30)), es.endP, nearestEdge, names.get( (articles.size()+2)%(names.size()-1))));
        } else {
          es.comments.add(new comment(new PVector(random(es.endP.x, es.startP.x), random(es.endP.y, es.startP.y), es.startP.z+ random(30)), es.startP, nearestEdge, names.get( (articles.size()+2)%(names.size()-1))));
        }
      }
      a.attach = es;

      if (! ways.contains(es)) {
        base.remove(es);
        ways.add(es);
      }
    } else if (nearestEdge == 3) {
      way es = new way(new PVector((i+1)*interval - crosswidth, (j+1)*interval - crosswidth, numZ), new PVector(i *interval+ crosswidth, (j+1)*interval- crosswidth, numZ));
      for (int t=0; t< es.commentNum; t++) {
        if (random(1)>0.5) {
          es.comments.add(new comment(new PVector(random(es.endP.x, es.startP.x), random(es.endP.y, es.startP.y), es.startP.z+ random(30)), es.endP, nearestEdge, names.get( (articles.size()+2)%(names.size()-1))));
        } else {
          es.comments.add(new comment(new PVector(random(es.endP.x, es.startP.x), random(es.endP.y, es.startP.y), es.startP.z+ random(30)), es.startP, nearestEdge, names.get( (articles.size()+2)%(names.size()-1))));
        }
      }
      a.attach = es;     

      if (! ways.contains(es)) {
        base.remove(es);
        ways.add(es);
      }
    }
  } else if (counter == AllI) {
    for (article a : articles) {


      a.attribute();
    }
  }

  for (article a : articles) {
    a.drawLine(articles.get(a.relationIndex));
    a.show();
    if (counter>AllI) {
      a.showText();
    }
  }
  if (counter>AllI) { 
    for (way a : ways) {
      for (comment mm : a.comments) {
        mm.showText();
      }
    }
  }

  angle += 0.02;
  counter += 1;

  popMatrix();
}



void keyPressed() {
  if (keyCode == ENTER)
  {
    textMode(SHAPE);
  }
}
