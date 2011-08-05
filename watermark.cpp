#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

string decode(Mat src){
  int wstep = src.cols / 8;
  int hstep = src.rows / 8;
  int chsize = wstep * hstep / 8;
  char *cstr = new char[chsize];
  int chptr=0;
  char ch=0;
  
  for(int hidx=0;hidx<hstep-1;hidx++){
    for(int widx=0;widx<wstep-1;widx++){
      int sum[3] = {0,0,0};
      for(int y=hidx*8;y<(hidx+1)*8;y++){
        for(int x=widx*8;x<(widx+1)*8;x++){
          for(int k=0;k<3;k++){
            sum[k] += (uchar)src.data[y*src.step+x*3+k];
          }
        }
      }
      int tf=0;
      for(int k=0;k<3;k++){
        int d =  sum[k]/64;
        tf += (d & 0x0000000000000004)>>2;
      }
      int code = (tf<=1)? 0 : 1;
      int base = chptr / 8;
      int offset = chptr % 8;
      if( offset==0 ){
        ch = code;
      }else{
        ch |= (code << offset);
      }
      if( offset==7 ){
        cstr[base] = ch;
      }
      chptr++;
    }
  }
  cstr[chsize-1] = 0;
  return string(cstr);
}


bool encode(Mat src, Mat dst, string str){
  int wstep = src.cols / 8;
  int hstep = src.rows / 8;
  int chsize = str.size();
  if( str.size()-1 > (wstep*hstep/8) ){
    return false;
  }
  const char* cstr = str.c_str();
  
  // init
  for(int y=0;y<src.rows;y++){
    for(int x=0;x<src.cols;x++){
      for(int k=0;k<3;k++){
        int data = (uchar)src.data[y*src.step + x*3 + k];
        data = (data>=251) ? 251 : data;
        data = (data<=0) ? 0 : data;
        dst.data[y*src.step+x*3+k] = (uchar)data;
      }
    }
  }
  
  int chptr = 0;
  for(int hidx = 0; hidx<hstep-1; hidx++){
    for(int widx=0; widx<wstep-1; widx++){
      int sum[3] = {0,0,0};
      for(int y=hidx*8; y<(hidx+1)*8;y++){
        for(int x=widx*8; x<(widx+1)*8;x++){
          for(int k=0;k<3;k++){
            sum[k] += (uchar)src.data[y*src.step+x*3+k];
          }
        }
      }
      int rest[3];
      for(int i=0;i<3;i++){
        rest[i] = sum[i] % 64;
        sum[i] = sum[i]/64;
      }
      int base = chptr / 8;
      int offset = chptr % 8;
      int code = (cstr[base] >> offset) & 0x0000000000000001;
      for(int k=0;k<3;k++){
        int ccode = (sum[k] & 0x0000000000000004)>>2;
        int v = (code!=ccode)? 4 : 0;
        // decrement
        if( v==0 ){ continue; }
        for(int y=hidx*8;y<(hidx+1)*8;y++){
          for(int x=widx*8; x<(widx+1)*8;x++){
            dst.data[y*dst.step+x*3+k]+=(uchar)v;
          }
        }
      }
      chptr++;
    }
  }
  return true;
}

int dpmatch(int *dy, int ny, int *dx, int nx){
  int map[ny*nx];
  int dir[ny*nx];
  // enter penality
  for(int y=0;y<ny;y++){
    for(int x=0;x<nx;x++){
      if( dy[y] == dx[x] ){
        map[y*nx+x] = 0;
      }else{
        map[y*nx+x] = 3;
      }
    }
  }
  dir[0] = 0;
  const int DOWN = 0;
  const int LEFT = 1;
  const int LEFTDOWN = 2;
  for(int y=1;y<ny;y++){
    dir[y*nx] = DOWN;
    map[y*nx] = map[(y-1)*nx] + map[y*nx] + 1;
  }
  for(int x=1;x<nx;x++){
    dir[x] = LEFT;
    map[x] = map[x-1] + map[x] + 1;
  }
  
  for(int y=1;y<ny;y++){
    for(int x=1;x<nx;x++){
      int cost_down = map[(y-1)*nx+x] + 1 + map[y*nx+x];
      int cost_left = map[y*nx+x-1] + 1 + map[y*nx+x];
      int cost_leftdown = map[(y-1)*nx+x-1] + map[y*nx+x];
      if( cost_leftdown <= cost_left ){
        if( cost_leftdown <= cost_down ){
          map[y*nx+x] = cost_leftdown;
          dir[y*nx+x] = LEFTDOWN;
        }else{
          map[y*nx+x] = cost_down;
          dir[y*nx+x] = DOWN;
        }
      }else{
        if( cost_left <= cost_down ){
          map[y*nx+x] = cost_left;
          dir[y*nx+x] = LEFT;
        }else{
          map[y*nx+x] = cost_down;
          dir[y*nx+x] = DOWN;
        }
      }
    }
  }
  return map[ny*nx-1];
}

string decode_sub(Mat src){
  int wstep = src.cols / 4;
  int hstep = src.rows / 4;
  int chsize = wstep * hstep / 8;
  char *cstr = new char[chsize];
  int chptr=0;
  char ch=0;
  
  // 0  1  5  6
  // 2  4  7 12
  // 3  8 11 13
  // 9 10 14 15
  int xorder[] = {0,1,0,0, 1,2,3,2, 1,0,1,2, 3,3,2,3};
  int yorder[] = {0,0,1,2, 1,0,0,1, 2,3,3,2, 1,2,3,3};
  // 1 => 1100
  // 0 => 0011
  for(int hidx=0;hidx<hstep-1;hidx++){
    for(int widx=0;widx<wstep-1;widx++){
      int ref0[16] = {0,0,1,1, 0,0,1,1, 0,0,1,1, 0,0,1,1};
      int ref1[16] = {1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0};
      int cmp[16];
      int tf=0;
      int bx = widx*4;
      int by = widx*4;
      for(int k=0;k<3;k++){
        for(int i=0;i<16;i++){
          cmp[i] = src.data[(yorder[i]+by)*src.step+(xorder[i]+bx)*3+k] & 0x0001;
        }
        int v0 = dpmatch(cmp,16,ref0,16);
        int v1 = dpmatch(cmp,16,ref1,16);
        if( v1<=v0 ){ tf++; }
      }
      int code = (tf<=1)? 0 : 1;
      int base = chptr / 8;
      int offset = chptr % 8;
      printf("%2d",code);
      if( offset==0 ){
        ch = code;
      }else{
        ch |= (code << offset);
      }
      if( offset==7 ){
        cstr[base] = ch;
      }
      chptr++;
    }
  }
  printf("\n");
  cstr[chsize-1] = 0;
  return string(cstr);
}

bool encode_sub(Mat src, Mat dst, string str){
  int wstep = src.cols / 4;
  int hstep = src.rows / 4;
  int chsize = str.size();
  if( str.size()-1 > (wstep*hstep/8) ){
    return false;
  }
  const char* cstr = str.c_str();
  // init
  for(int y=0;y<src.rows;y++){
    for(int x=0;x<src.cols;x++){
      for(int k=0;k<3;k++){
        int data = (uchar)src.data[y*src.step + x*3 + k];
        //data = (data>=254) ? 254 : data;
        //data = (data<=0) ? 0 : data;
        dst.data[y*src.step+x*3+k] = (uchar)data;
      }
    }
  }
  
  // 0  1  5  6
  // 2  4  7 12
  // 3  8 11 13
  // 9 10 14 15
  int xorder[] = {0,1,0,0, 1,2,3,2, 1,0,1,2, 3,3,2,3};
  int yorder[] = {0,0,1,2, 1,0,0,1, 2,3,3,2, 1,2,3,3};
  // 1 => 1100
  // 0 => 0011
  int chptr = 0;
  for(int hidx = 0; hidx<hstep-1; hidx++){
    for(int widx=0; widx<wstep-1; widx++){
      int base = chptr / 8;
      int offset = chptr % 8;
      int code = (cstr[base] >> offset) & 0x0000000000000001;
      int bx = widx*4;
      int by = hidx*4;
      if( code == 1 ){
        for(int k=0;k<3;k++){
          for(int i=0;i<16;i+=4){
            // 1100
            dst.data[(yorder[i]+by)*dst.step+(xorder[i]+bx)*3+k] |= 0x0001;
            dst.data[(yorder[i+1]+by)*dst.step+(xorder[i+1]+bx)*3+k] |= 0x0001;
            dst.data[(yorder[i+2]+by)*dst.step+(xorder[i+2]+bx)*3+k] &= 0xfffe;
            dst.data[(yorder[i+3]+by)*dst.step+(xorder[i+3]+bx)*3+k] &= 0xfffe;
          }
        }
      }else{
        for(int k=0;k<3;k++){
          for(int i=0;i<16;i+=4){
            // 0011
            dst.data[(yorder[i]+by)*dst.step+(xorder[i]+bx)*3+k] &= 0xfffe;
            dst.data[(yorder[i+1]+by)*dst.step+(xorder[i+1]+bx)*3+k] &= 0xfffe;
            dst.data[(yorder[i+2]+by)*dst.step+(xorder[i+2]+bx)*3+k] |= 0x0001;
            dst.data[(yorder[i+3]+by)*dst.step+(xorder[i+3]+bx)*3+k] |= 0x0001;
          }
        }
      }
      chptr++;
    }
  }
  return true;
}



void usage(char *cmd){
  cout << "usage:" << endl;
  cout << cmd << " <input file> (<encode string>)" << endl;
  cout << "ex1. encode string " << endl;
  cout << "\t" << cmd << " decode_to_encode.jpg informations" << endl;
  cout << "ex2. decode string " << endl;
  cout << "\t" << cmd << " input.jpg" << endl;
}

int main(int argc,char** argv){
  char *infile = (argc>1)? argv[1] : NULL;
  char *encode_string = (argc>2)? argv[2] : NULL;
  bool isencode = (argc>2)? true : false;
  if( infile == NULL){
    usage(argv[0]);
    return 0;
  }
  
  Mat src = imread(infile,1); // force to read 
  
  if( isencode ){
    if( !encode_sub(src,src,encode_string) ){
      cerr << "overflow string.." << endl;
      int len = strlen(encode_string);
      int limit = src.rows*src.cols/64-1;
      if( len>limit ){
        cerr << "limit:" << limit << endl;
        cerr << "len:" << len << endl;
      }
    }else{
      imwrite(infile,src);
    }
  }else{
    string str = decode_sub(src);
    cout << str << endl;
  }
  
  return 0;
}
