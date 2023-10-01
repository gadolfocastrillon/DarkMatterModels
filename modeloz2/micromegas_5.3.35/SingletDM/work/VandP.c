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
, {"l","L",0, 15, "Ml","0",1,1,2,-3}
, {"u","U",0, 2, "Mu","0",1,3,6,2}
, {"d","D",0, 1, "Md","0",1,3,6,-1}
, {"c","C",0, 4, "Mc","0",1,3,6,2}
, {"s","S",0, 3, "Ms","0",1,3,6,-1}
, {"t","T",0, 6, "Mt","wtop",1,3,6,2}
, {"b","B",0, 5, "Mb","0",1,3,6,-1}
, {"h","h",1, 25, "Mh","wh",0,1,1,0}
, {"~x1","~x1",1, 35, "Mdm1","wX1",0,1,1,0}
};
ModelPrtclsStr *ModelPrtcls=ModelPrtcls_; 
int nModelVars=20;
int nModelFunc=51;
static int nCurrentVars=19;
int*currentVarPtr=&nCurrentVars;
static char*varNames_[71]={
 "EE","alfSMZ","SW","MZ","Q","Mtp","MbMb","McMc","wZ","wW"
,"Mm","Ml","Mu","Md","Ms","wtop","Mh","laS","laSH","Mdm1"
,"CW","MW","Lqcd","Mb","Mt","Mc","PI","Mbp","Mcp","V"
,"muSq1","aQCD","rhF_c","ihF_c","rhF_b","ihF_b","rhF_t","ihF_t","rhF_l","ihF_l"
,"rhV_W","ihV_W","McR","MbR","MtR","rhF1_c","ihF1_c","rhF1_b","ihF1_b","rhF1_t"
,"ihF1_t","rhF1_l","ihF1_l","ahF_c","ahF_b","ahF_t","ahF_l","a_hV_W","Rqcd","lnTop"
,"Ctop","Cq","alphaE0","Qu","Qd","LGGH","LAAH","aSMhF_f","aSMhV_W","LGGSM"
,"LAASM"};
char**varNames=varNames_;
static REAL varValues_[71]={
   3.122300E-01,  1.184000E-01,  4.810000E-01,  9.118700E+01,  1.000000E+02,  1.735000E+02,  4.230000E+00,  1.270000E+00,  2.502000E+00,  2.094000E+00
,  1.057000E-01,  1.777000E+00,  1.000000E-02,  1.000000E-02,  2.000000E-01,  1.442000E+00,  1.250000E+02,  2.000000E-01,  1.000000E-01,  5.000000E+01
};
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
   nCurrentVars=20;
   V[20]=Sqrt(1-Pow(V[2],2));
   if(!isfinite(V[20]) || FError) return 20;
   nCurrentVars=21;
   V[21]=V[3]*V[20];
   if(!isfinite(V[21]) || FError) return 21;
   nCurrentVars=22;
   V[22]=initQCD5(V[1],V[7],V[6],V[5]);
   if(!isfinite(V[22]) || FError) return 22;
 FirstQ:
 cErr=1;
   nCurrentVars=23;
   V[23]=MbEff(V[4]);
   if(!isfinite(V[23]) || FError) return 23;
   nCurrentVars=24;
   V[24]=MtEff(V[4]);
   if(!isfinite(V[24]) || FError) return 24;
   nCurrentVars=25;
   V[25]=McEff(V[4]);
   if(!isfinite(V[25]) || FError) return 25;
   nCurrentVars=26;
   V[26]=Acos(-1);
   if(!isfinite(V[26]) || FError) return 26;
   nCurrentVars=27;
   V[27]=V[6]*(1+4/(double)((3))*alphaQCD(V[6])/(V[26]));
   if(!isfinite(V[27]) || FError) return 27;
   nCurrentVars=28;
   V[28]=V[25]*(1+4/(double)((3))*alphaQCD(V[25])/(V[26]));
   if(!isfinite(V[28]) || FError) return 28;
   nCurrentVars=29;
   V[29]=2*V[21]/(V[0])*V[2];
   if(!isfinite(V[29]) || FError) return 29;
   nCurrentVars=30;
   V[30]=Pow(V[19],2)-V[18]*Pow(V[29],2);
   if(!isfinite(V[30]) || FError) return 30;
   nCurrentVars=31;
   V[31]=alphaQCD(V[4])/(V[26]);
   if(!isfinite(V[31]) || FError) return 31;
   nCurrentVars=32;
   V[32]=Creal(HggF(Pow(V[4]/(2)/(V[28]),2)));
   if(!isfinite(V[32]) || FError) return 32;
   nCurrentVars=33;
   V[33]=Cimag(HggF(Pow(V[4]/(2)/(V[28]),2)));
   if(!isfinite(V[33]) || FError) return 33;
   nCurrentVars=34;
   V[34]=Creal(HggF(Pow(V[4]/(2)/(V[27]),2)));
   if(!isfinite(V[34]) || FError) return 34;
   nCurrentVars=35;
   V[35]=Cimag(HggF(Pow(V[4]/(2)/(V[27]),2)));
   if(!isfinite(V[35]) || FError) return 35;
   nCurrentVars=36;
   V[36]=Creal(HggF(Pow(V[4]/(2)/(V[5]),2)));
   if(!isfinite(V[36]) || FError) return 36;
   nCurrentVars=37;
   V[37]=Cimag(HggF(Pow(V[4]/(2)/(V[5]),2)));
   if(!isfinite(V[37]) || FError) return 37;
   nCurrentVars=38;
   V[38]=Creal(HggF(Pow(V[4]/(2)/(V[11]),2)));
   if(!isfinite(V[38]) || FError) return 38;
   nCurrentVars=39;
   V[39]=Cimag(HggF(Pow(V[4]/(2)/(V[11]),2)));
   if(!isfinite(V[39]) || FError) return 39;
   nCurrentVars=40;
   V[40]=Creal(HggV(Pow(V[4]/(2)/(V[21]),2)));
   if(!isfinite(V[40]) || FError) return 40;
   nCurrentVars=41;
   V[41]=Cimag(HggV(Pow(V[4]/(2)/(V[21]),2)));
   if(!isfinite(V[41]) || FError) return 41;
   nCurrentVars=42;
   V[42]=V[28]*McRun(V[4]/(2))/(McRun(V[28]));
   if(!isfinite(V[42]) || FError) return 42;
   nCurrentVars=43;
   V[43]=V[27]*MbRun(V[4]/(2))/(MbRun(V[27]));
   if(!isfinite(V[43]) || FError) return 43;
   nCurrentVars=44;
   V[44]=V[5]*MtRun(V[4]/(2))/(MtRun(V[5]));
   if(!isfinite(V[44]) || FError) return 44;
   nCurrentVars=45;
   V[45]=Creal(HggF(Pow(V[4]/(2)/(V[42]),2))*(1+V[31]*Hgam1F(Pow(V[4]/(2)/(V[42]),2))));
   if(!isfinite(V[45]) || FError) return 45;
   nCurrentVars=46;
   V[46]=Cimag(HggF(Pow(V[4]/(2)/(V[42]),2))*(1+V[31]*Hgam1F(Pow(V[4]/(2)/(V[42]),2))));
   if(!isfinite(V[46]) || FError) return 46;
   nCurrentVars=47;
   V[47]=Creal(HggF(Pow(V[4]/(2)/(V[43]),2))*(1+V[31]*Hgam1F(Pow(V[4]/(2)/(V[43]),2))));
   if(!isfinite(V[47]) || FError) return 47;
   nCurrentVars=48;
   V[48]=Cimag(HggF(Pow(V[4]/(2)/(V[43]),2))*(1+V[31]*Hgam1F(Pow(V[4]/(2)/(V[43]),2))));
   if(!isfinite(V[48]) || FError) return 48;
   nCurrentVars=49;
   V[49]=Creal(HggF(Pow(V[4]/(2)/(V[44]),2))*(1+V[31]*Hgam1F(Pow(V[4]/(2)/(V[44]),2))));
   if(!isfinite(V[49]) || FError) return 49;
   nCurrentVars=50;
   V[50]=Cimag(HggF(Pow(V[4]/(2)/(V[44]),2))*(1+V[31]*Hgam1F(Pow(V[4]/(2)/(V[44]),2))));
   if(!isfinite(V[50]) || FError) return 50;
   nCurrentVars=51;
   V[51]=Creal(HggF(Pow(V[4]/(2)/(V[11]),2))*(1+V[31]*Hgam1F(Pow(V[4]/(2)/(V[11]),2))));
   if(!isfinite(V[51]) || FError) return 51;
   nCurrentVars=52;
   V[52]=Cimag(HggF(Pow(V[4]/(2)/(V[11]),2))*(1+V[31]*Hgam1F(Pow(V[4]/(2)/(V[11]),2))));
   if(!isfinite(V[52]) || FError) return 52;
   nCurrentVars=53;
   V[53]=-V[0]/(V[21])*V[25]/(V[2])/(2)/(V[25]);
   if(!isfinite(V[53]) || FError) return 53;
   nCurrentVars=54;
   V[54]=-V[0]/(V[21])*V[23]/(V[2])/(2)/(V[23]);
   if(!isfinite(V[54]) || FError) return 54;
   nCurrentVars=55;
   V[55]=-V[0]/(V[21])*V[24]/(V[2])/(2)/(V[24]);
   if(!isfinite(V[55]) || FError) return 55;
   nCurrentVars=56;
   V[56]=-V[0]/(V[21])*V[11]/(V[2])/(2)/(V[11]);
   if(!isfinite(V[56]) || FError) return 56;
   nCurrentVars=57;
   V[57]=V[0]*V[21]/(V[2])/(Pow(V[21],2))/(2);
   if(!isfinite(V[57]) || FError) return 57;
   nCurrentVars=58;
   V[58]=1+V[31]*(149/(double)((12))+V[31]*(68.6482-V[31]*212.447));
   if(!isfinite(V[58]) || FError) return 58;
   nCurrentVars=59;
   V[59]=2*Log(V[5]/(V[4]));
   if(!isfinite(V[59]) || FError) return 59;
   nCurrentVars=60;
   V[60]=1+V[31]*(11/(double)((4))+V[31]*(6.1537-2.8542*V[59]+V[31]*(10.999-17.93*V[59]+5.47*Pow(V[59],2))));
   if(!isfinite(V[60]) || FError) return 60;
   nCurrentVars=61;
   V[61]=1+11/(double)((4))*V[31];
   if(!isfinite(V[61]) || FError) return 61;
   nCurrentVars=62;
   V[62]=1/(137.036);
   if(!isfinite(V[62]) || FError) return 62;
   nCurrentVars=63;
   V[63]=2/(double)((3));
   if(!isfinite(V[63]) || FError) return 63;
   nCurrentVars=64;
   V[64]=-1/(double)((3));
   if(!isfinite(V[64]) || FError) return 64;
   nCurrentVars=65;
   V[65]=-V[31]/(8)*1/(double)((2))*Sqrt(V[58])*Cabs(((V[34]+I*V[35])*V[54]+(V[32]+I*V[33])*V[53])*V[61]+(V[36]+I*V[37])*V[55]*V[60]);
   if(!isfinite(V[65]) || FError) return 65;
   nCurrentVars=66;
   V[66]=-V[62]/(8*V[26])*Cabs(3*Pow(V[64],2)*(V[47]+I*V[48])*V[54]+3*Pow(V[63],2)*((V[49]+I*V[50])*V[55]+(V[45]+I*V[46])*V[53])+(V[38]+I*V[39])*V[56]-(V[40]+I*V[41])*V[57]);
   if(!isfinite(V[66]) || FError) return 66;
   nCurrentVars=67;
   V[67]=-V[0]/(V[21])/(V[2])/(2);
   if(!isfinite(V[67]) || FError) return 67;
   nCurrentVars=68;
   V[68]=V[0]/(V[2])/(V[21])/(2);
   if(!isfinite(V[68]) || FError) return 68;
   nCurrentVars=69;
   V[69]=-V[31]/(8)*1/(double)((2))*Sqrt(V[58])*Fabs(V[67])*Cabs((V[34]+I*V[35]+V[32]+I*V[33])*V[61]+(V[36]+I*V[37])*V[60]);
   if(!isfinite(V[69]) || FError) return 69;
   nCurrentVars=70;
   V[70]=-V[62]/(8*V[26])*Cabs(3*Pow(V[64],2)*(V[47]+I*V[48])*V[67]+3*Pow(V[63],2)*((V[49]+I*V[50])*V[67]+(V[45]+I*V[46])*V[67])+(V[38]+I*V[39])*V[67]-(V[40]+I*V[41])*V[68]);
   if(!isfinite(V[70]) || FError) return 70;
   if(VV==NULL) 
   {  VV=malloc(sizeof(REAL)*nModelVars);
      for(i=0;i<nModelVars;i++) if(strcmp(varNames[i],"Q")==0) iQ=i;
   }
   for(i=0;i<nModelVars;i++) VV[i]=V[i];
   cErr=0;
   nCurrentVars++;
   return 0;
}
