#include <string>
#include <fstream>
#include <iostream>
#include <TMath.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TLatex.h>
#include <TLine.h>
void vXenon(void)
{
 TCanvas *Can = new TCanvas("c","velocity distribution uncertainty for Xenon",200,10,600,400);
 double xMin=5.477226E+00, xMax=1.083470E+03, yMin=2.000000E-47, yMax=6.000000E-44;
 char buff[2000];
 double  X0[29],dX0[29], X1[29],dX1[29], X2[29],dX2[29], X3[29],dX3[29];
 Can->SetLogx();
 for(int i=0;i<29;i++){ X0[i]=xMin*pow(xMax/xMin,(i+0.5)/29);dX0[i]=X0[i]*(pow(xMax/xMin,0.333/29)-1);} 
 for(int i=0;i<29;i++){ X1[i]=xMin*pow(xMax/xMin,(i+0.5)/29);dX1[i]=X1[i]*(pow(xMax/xMin,0.333/29)-1);} 
 for(int i=0;i<29;i++){ X2[i]=xMin*pow(xMax/xMin,(i+0.5)/29);dX2[i]=X2[i]*(pow(xMax/xMin,0.333/29)-1);} 
 for(int i=0;i<29;i++){ X3[i]=xMin*pow(xMax/xMin,(i+0.5)/29);dX3[i]=X3[i]*(pow(xMax/xMin,0.333/29)-1);} 
 double Xtext = xMin*pow(xMax/xMin,0.3);
 double Ytext= yMax;
 Can->SetLogy();
 double dYtext=pow(yMax/yMin,0.055);
 TH1F *hr = Can->DrawFrame(xMin,yMin,xMax,yMax);
 hr->SetXTitle("Mcdm");
 double  Y0[29];
 double  Y1[29];
 double  Y2[29];
 double  Y3[29];
 FILE*f=fopen("vXenon.tab","r");
 for(int i=0;i<29;)
 {
  fscanf(f,"%[^\n]%*c",buff);
  if(buff[0]!='#') {  sscanf(buff," %lf %lf %lf %lf"  ,Y0+i ,Y1+i ,Y2+i ,Y3+i); i++;    }
 }
 fclose(f);
 TLatex ltx;
 ltx.SetTextFont(42);
 ltx.SetTextSize(0.04);
int i0,i1;
   for(i0=0;!isfinite(Y0[i0]);i0++);
   for(i1=28;!isfinite(Y0[i1]);i1--);
   TGraph *gr0 = new TGraph (1+i1-i0,X0+i0,Y0+i0);
   gr0->SetLineColor(1);
   gr0->Draw("L");
   Ytext/=dYtext;
    ltx.SetTextColor(1);
    ltx.DrawLatex(Xtext,Ytext,"std");
   for(i0=0;!isfinite(Y1[i0]);i0++);
   for(i1=28;!isfinite(Y1[i1]);i1--);
   TGraph *gr1 = new TGraph (1+i1-i0,X1+i0,Y1+i0);
   gr1->SetLineColor(2);
   gr1->Draw("L");
   Ytext/=dYtext;
    ltx.SetTextColor(2);
    ltx.DrawLatex(Xtext,Ytext,"min");
   for(i0=0;!isfinite(Y2[i0]);i0++);
   for(i1=28;!isfinite(Y2[i1]);i1--);
   TGraph *gr2 = new TGraph (1+i1-i0,X2+i0,Y2+i0);
   gr2->SetLineColor(3);
   gr2->Draw("L");
   Ytext/=dYtext;
    ltx.SetTextColor(3);
    ltx.DrawLatex(Xtext,Ytext,"max");
   for(i0=0;!isfinite(Y3[i0]);i0++);
   for(i1=28;!isfinite(Y3[i1]);i1--);
   TGraph *gr3 = new TGraph (1+i1-i0,X3+i0,Y3+i0);
   gr3->SetLineColor(4);
   gr3->Draw("L");
   Ytext/=dYtext;
    ltx.SetTextColor(4);
    ltx.DrawLatex(Xtext,Ytext,"SHM++");
 Can->Print("vXenon.pdf");
}
