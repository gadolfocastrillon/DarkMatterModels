#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../CalcHEP_src/include/extern.h"
#include "../../CalcHEP_src/include/VandP.h"
#include "autoprot.h"
extern int  FError;
/*  Special model functions  */

int nModelParticles=19;
static ModelPrtclsStr ModelPrtcls_[19]=
{
  {"A","A",1, 22, "0","0",2,1,2,0}
, {"Z","Z",1, 23, "MZ","wZ",2,1,3,0}
, {"G","G",1, 21, "0","0",2,8,16,0}
, {"W+","W-",0, 24, "MW","wW",2,1,3,3}
, {"n1","N1",0, 12, "0","0",1,1,1,0}
, {"e1","E1",0, 11, "0","0",1,1,2,-3}
, {"n2","N2",0, 14, "0","0",1,1,1,0}
, {"e2","E2",0, 13, "Mm","0",1,1,2,-3}
, {"n3","N3",0, 16, "0","0",1,1,1,0}
, {"e3","E3",0, 15, "Mt","0",1,1,2,-3}
, {"u","U",0, 2, "Mup","0",1,3,6,2}
, {"d","D",0, 1, "Md","0",1,3,6,-1}
, {"c","C",0, 4, "Mc","0",1,3,6,2}
, {"s","S",0, 3, "Ms","0",1,3,6,-1}
, {"t","T",0, 6, "Mtop","wtop",1,3,6,2}
, {"b","B",0, 5, "Mb","0",1,3,6,-1}
, {"h","h",1, 25, "MH","wh",0,1,1,0}
, {"~~s2","~~s2",1, 1002, "Ms2","0",0,1,1,0}
, {"~p1","~P1",0, 1001, "Mp1","0",1,1,2,0}
};
ModelPrtclsStr *ModelPrtcls=ModelPrtcls_; 
int nModelVars=24;
int nModelFunc=14;
static int nCurrentVars=23;
int*currentVarPtr=&nCurrentVars;
static char*varNames_[38]={
 "EE","SW","s12","s23","s13","MZ","wZ","wW","Mm","Mt"
,"Mup","Md","Mc","Ms","Mtop","wtop","Mb","MH","Ms2","Mp1"
,"laS","ys","yp","vEW","CW","c12","c23","c13","Vud","Vus"
,"Vub","Vcd","Vcs","Vcb","Vtd","Vts","Vtb","MW"};
char**varNames=varNames_;
static REAL varValues_[38]={
   3.133300E-01,  4.740000E-01,  2.210000E-01,  4.000000E-02,  3.500000E-03,  9.118700E+01,  2.502000E+00,  2.094000E+00,  1.057000E-01,  1.777000E+00
,  5.000000E-02,  1.000000E-02,  1.300000E+00,  2.000000E-01,  1.700000E+02,  1.442000E+00,  4.300000E+00,  1.260000E+02,  9.000000E+02,  4.000000E+02
,  1.000000E-01,  5.000000E-01,  5.000000E-01,  2.460000E+02};
REAL*varValues=varValues_;
int calcMainFunc(void)
{
   int i;
   static REAL * VV=NULL;
   static int iQ=-1;
   static int cErr=1;
   REAL *V=varValues;
   FError=0;
   if(VV && cErr==0)
   { for(i=0;i<nModelVars;i++) if(i!=iQ && VV[i]!=V[i]) break;
     if(i==nModelVars)     return 0;
   }
  cErr=1;
   nCurrentVars=24;
   V[24]=Sqrt(1-Pow(V[1],2));
   if(!isfinite(V[24]) || FError) return 24;
   nCurrentVars=25;
   V[25]=Sqrt(1-Pow(V[2],2));
   if(!isfinite(V[25]) || FError) return 25;
   nCurrentVars=26;
   V[26]=Sqrt(1-Pow(V[3],2));
   if(!isfinite(V[26]) || FError) return 26;
   nCurrentVars=27;
   V[27]=Sqrt(1-Pow(V[4],2));
   if(!isfinite(V[27]) || FError) return 27;
   nCurrentVars=28;
   V[28]=V[25]*V[27];
   if(!isfinite(V[28]) || FError) return 28;
   nCurrentVars=29;
   V[29]=V[2]*V[27];
   if(!isfinite(V[29]) || FError) return 29;
   nCurrentVars=30;
   V[30]=V[4];

   nCurrentVars=31;
   V[31]=-V[25]*V[3]*V[4]-V[2]*V[26];
   if(!isfinite(V[31]) || FError) return 31;
   nCurrentVars=32;
   V[32]=V[25]*V[26]-V[2]*V[3]*V[4];
   if(!isfinite(V[32]) || FError) return 32;
   nCurrentVars=33;
   V[33]=V[3]*V[27];
   if(!isfinite(V[33]) || FError) return 33;
   nCurrentVars=34;
   V[34]=V[2]*V[3]-V[25]*V[26]*V[4];
   if(!isfinite(V[34]) || FError) return 34;
   nCurrentVars=35;
   V[35]=-V[2]*V[26]*V[4]-V[25]*V[3];
   if(!isfinite(V[35]) || FError) return 35;
   nCurrentVars=36;
   V[36]=V[26]*V[27];
   if(!isfinite(V[36]) || FError) return 36;
   nCurrentVars=37;
   V[37]=V[5]*V[24];
   if(!isfinite(V[37]) || FError) return 37;
   if(VV==NULL) 
   {  VV=malloc(sizeof(REAL)*nModelVars);
      for(i=0;i<nModelVars;i++) if(strcmp(varNames[i],"Q")==0) iQ=i;
   }
   for(i=0;i<nModelVars;i++) VV[i]=V[i];
   cErr=0;
   nCurrentVars++;
   return 0;
}
