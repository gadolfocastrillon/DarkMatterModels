#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../CalcHEP_src/include/extern.h"
#include "../../CalcHEP_src/include/VandP.h"
#include "autoprot.h"
extern int  FError;
/*  Special model functions  */

int nModelParticles=18;
static ModelPrtclsStr ModelPrtcls_[18]=
{
  {"A","A",1, 22, "0","0",2,1,2,0}
, {"Z","Z",1, 23, "MZ","wZ",2,1,3,0}
, {"G","G",1, 21, "0","0",2,8,16,0}
, {"W+","W-",0, 24, "MW","wW",2,1,3,3}
, {"ne","Ne",0, 12, "0","0",1,1,1,0}
, {"e","E",0, 11, "0","0",1,1,2,-3}
, {"nm","Nm",0, 14, "0","0",1,1,1,0}
, {"m","M",0, 13, "Mm","0",1,1,2,-3}
, {"nl","Nl",0, 16, "0","0",1,1,1,0}
, {"l","L",0, 15, "Mt","0",1,1,2,-3}
, {"u","U",0, 2, "Mu","0",1,3,6,2}
, {"d","D",0, 1, "Md","0",1,3,6,-1}
, {"c","C",0, 4, "Mc","0",1,3,6,2}
, {"s","S",0, 3, "Ms","0",1,3,6,-1}
, {"t","T",0, 6, "Mtop","wtop",1,3,6,2}
, {"b","B",0, 5, "Mb","0",1,3,6,-1}
, {"h","h",1, 25, "Mh","wh",0,1,1,0}
, {"~phi1","~PHI1",0, 50, "Mp1","wHP",0,1,1,0}
};
ModelPrtclsStr *ModelPrtcls=ModelPrtcls_; 
int nModelVars=19;
int nModelFunc=3;
static int nCurrentVars=18;
int*currentVarPtr=&nCurrentVars;
static char*varNames_[22]={
 "EE","SW","MZ","Q","Mp1","laphi","laSH1","mu32","wZ","wW"
,"Mm","Mt","Mu","Md","Mc","Ms","Mtop","wtop","Mh","CW"
,"MW","Mb"};
char**varNames=varNames_;
static REAL varValues_[22]={
   3.122300E-01,  4.810000E-01,  9.118700E+01,  1.000000E+02,  2.000000E+02,  1.000000E-02,  1.000000E-02,  1.000000E+03,  2.502000E+00,  2.094000E+00
,  1.057000E-01,  1.777000E+00,  1.000000E-02,  1.000000E-02,  1.300000E+00,  2.000000E-01,  1.700000E+02,  1.442000E+00,  2.000000E+02};
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
     if(i==nModelVars)      {if(iQ>=0 && VV[iQ]!=V[iQ]) goto FirstQ; else return 0;} 
   }
  cErr=1;
   nCurrentVars=19;
   V[19]=Sqrt(1-Pow(V[1],2));
   if(!isfinite(V[19]) || FError) return 19;
   nCurrentVars=20;
   V[20]=V[2]*V[19];
   if(!isfinite(V[20]) || FError) return 20;
 FirstQ:
 cErr=1;
   nCurrentVars=21;
   V[21]=MbEff(V[3]);
   if(!isfinite(V[21]) || FError) return 21;
   if(VV==NULL) 
   {  VV=malloc(sizeof(REAL)*nModelVars);
      for(i=0;i<nModelVars;i++) if(strcmp(varNames[i],"Q")==0) iQ=i;
   }
   for(i=0;i<nModelVars;i++) VV[i]=V[i];
   cErr=0;
   nCurrentVars++;
   return 0;
}
