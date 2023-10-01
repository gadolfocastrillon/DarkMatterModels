
#include"../include/micromegas.h"
#include"../include/micromegas_aux.h"

static int N=0;
double p=-0.7;
double a=1E-5;
double c=0.5;
double z=0.1;
double A=10;
//double fun(double x) { N++; if(x>0.557)  return sin(x); else return 0;}

double funBW(double x)   { N++; return 1/( x*x +a*a);   } 
double funPole(double x) { N++; return  pow(fabs(x),p); }
double funPoleArg(double x, void*p) { double *pp=p; return  pow(fabs(x),*pp);};
double funRand(double x) { N++; return  0.5+ drand48(); }
double funStep(double x) { N++; if(x<z) return 1; else return 0;}
double funExp(double x)  { N++; return exp(A*x);        }
double funSin(double x)  { N++; return sin(x);          }

int main(int argc,char** argv)
{ int err,n,i;

  double ans;
N=0;  
  ans=simpson(funExp,-1,1,1E-3,NULL);
  printf("Exponent     A=%.2E  ans=%.3E delta=%.2E Ncall=%d\n\n", A,ans,  (exp(A)-exp(-A))/A/ans-1,N); 
N=0;  
  ans=simpson(funSin,-1,1,1E-3,NULL);
  printf("sin               ans=%.3E     Ncall=%d\n\n", ans,N ); 
      
N=0;  
  ans=simpson(funBW,-1,1,1E-3,NULL);
  printf("Breit-Wigner a=%.2E  ans=%.3E delta=%.2E Ncall=%d\n\n", a,ans, (2*atan(1/a)/a)/ans-1,N);    
/*
N=0;
  ans=simpson(funPole,-1,1,1E-3,NULL); 
  printf(" Pole        p=%.2E  ans=%.3E delta=%.2E Ncall=%d\n\n", p,ans,    2/(1+p)/ans-1,N); 
*/
N=0;
  ans=simpson_arg(funPoleArg, &p,-1,1,1E-3,NULL); 
  printf(" PoleArg     p=%.2E  ans=%.3E delta=%.2E Ncall=%d\n\n", p,ans,    2/(1+p)/ans-1,N); 
N=0;
  ans=simpson(funRand,-1,1,1E-3,NULL); 
  printf(" Random      c=%.2E  ans=%.3E delta=%.2E Ncall=%d\n\n", c,ans,    2*(c+0.5)/ans-1,N);    
N=0; 
  ans=simpson(funStep,-1,1,1E-3,NULL); 
  printf(" Step        z=%.2E  ans=%.3E delta=%.2E Ncall=%d\n\n", z,ans,    (1+z)/ans-1,N);  
}

