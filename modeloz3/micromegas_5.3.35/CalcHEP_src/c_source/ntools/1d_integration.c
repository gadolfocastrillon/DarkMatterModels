/*
 Copyright (C) 1997,2006, Alexander Pukhov 
*/
#include <math.h>
#include <stdio.h>
#include"1d_integration.h"
#include"rootDir.h"
#include"stdlib.h"

#define nErrMax 10

static double const X2[2]={2.113249E-01,7.886751E-01 };
static double const F2[2]={5.000000E-01,5.000000E-01 };
static double const X3[3]={1.127017E-01,5.000000E-01 ,8.872983E-01 };
static double const F3[3]={2.777778E-01,4.444444E-01 ,2.777778E-01 };
static double const X4[4]={6.943185E-02,3.300095E-01 ,6.699905E-01 ,9.305682E-01 };
static double const F4[4]={1.739274E-01,3.260726E-01 ,3.260726E-01 ,1.739274E-01 };
static double const X5[5]={4.691008E-02,2.307653E-01 ,5.000000E-01 ,7.692347E-01 ,9.530899E-01 };
static double const F5[5]={1.184634E-01,2.393143E-01 ,2.844445E-01 ,2.393143E-01 ,1.184634E-01 };
static double const X6[6]={3.376523E-02,1.693953E-01 ,3.806904E-01 ,6.193096E-01 ,8.306047E-01 ,9.662348E-01 };
static double const F6[6]={8.566223E-02,1.803808E-01 ,2.339570E-01 ,2.339570E-01 ,1.803808E-01 ,8.566225E-02 };
static double const X7[7]={2.544604E-02,1.292344E-01 ,2.970774E-01 ,5.000000E-01 ,7.029226E-01 ,8.707656E-01 ,9.745540E-01 };
static double const F7[7]={6.474248E-02,1.398527E-01 ,1.909150E-01 ,2.089796E-01 ,1.909150E-01 ,1.398527E-01 ,6.474248E-02 };



double gauss( double (*func)(double),double a,double b, int n)
{
        
  double ans=0;
  if(n<1) n=1;
  if(n>7) { printf(" 7 is a miximum number of points for Gauss integration (call with %d)\n",n); n=7;} 
  switch(n)
  {  int i;
    case 1: ans=(b-a)*func((a+b)/2);  break;
    case 2:
      for(i=0;i<n;i++) ans+=F2[i]*func(a+ (b-a)*X2[i]); break;
    case 3: 
      for(i=0;i<n;i++) ans+=F3[i]*func(a+ (b-a)*X3[i]); break;
    case 4:
      for(i=0;i<n;i++) ans+=F4[i]*func(a+ (b-a)*X4[i]); break;
    case 5:
      for(i=0;i<n;i++) ans+=F5[i]*func(a+ (b-a)*X5[i]); break;
    case 6:
      for(i=0;i<n;i++) ans+=F6[i]*func(a+ (b-a)*X6[i]); break;
    case 7:
      for(i=0;i<n;i++) ans+=F7[i]*func(a+ (b-a)*X7[i]); break;      
    default: 
      return 0;
  }
  return ans*(b-a);                       
 }



static void r_gauss( double(*func)(double),double a,double b, 
double eps, double * aEps, double * ans, double * aAns, int* N, int depth, int * err)
{
  int i,n;

  double s1,s2,s2a,s3,s3a,e_err,d=b-a;
  
  if(*N<0)     { *err=2; return;}
  if(depth>50) { *err=3;  printf("gauss345: depth>50 for [%e %e]\n", a,b);     return;}
  
  for(n=0,s1=0;n<3;n++) s1+=F3[n]*func(a+ d*X3[n]); s1*=d; *N-=3;
  for(n=0,s2=0,s2a=0;n<4;n++) {double  f=F4[n]*func(a+ d*X4[n]); s2+=f;s2a+=fabs(f);} 
  s2*=d; s2a*=fabs(d);*N-=4;
 
  if(!isfinite(s1) || ! isfinite(s2)) { *err=1; ; return;}  

  e_err=eps*s2a;
 
  if( fabs(s1-s2) <= 30*e_err)
  { 
    for(n=0,s3=0,s3a=0;n<5;n++) { double f=F5[n]*func(a+ d*X5[n]); s3+=f; s3a+=fabs(f); } 
    if(!isfinite(s3)) {  *err=1; return;} 
    s3*=d; s3a*=fabs(d); *N-=5;
    if(fabs(s3-s2) <= e_err) 
    { *ans+=s3;
      *aAns+=s3a;
      return;
    }
  }    

  if(fabs(s1-s2) <= 0.1*(*aEps)) 
  {  *ans+=s2;
     *aAns+=s2a;   
     *aEps -= fabs(s1-s2);
     return;
  }
         
  r_gauss(func,a,(a+b)/2,eps,aEps,ans,aAns,N,depth+1,err);
  if(*err) return;
  r_gauss(func,(a+b)/2,b,eps,aEps,ans,aAns,N,depth+1,err);
}   

double gauss345( double (*func)(double),double a,double b, double eps,int * err_code)
{
  double aEps; /* absolute error  */
  int n,k,err=0;	

  if(a==b) return 0;
  if(err_code) *err_code=0;

  for(n=0,aEps=0;n<4;n++) aEps+=F4[n]*fabs(func(a+ (b-a)*X4[n]));
  
  if(!isfinite(aEps)) { if(err_code) *err_code=1; else printf("gauss345: NaN in integrand\n"); 
                        return 0; 
                      }  
  if(aEps==0.)        return 0;
                      
  eps=eps/2;
  aEps = eps*aEps*fabs(b-a);


  for(k=0;;k++)
  {  double ans=0., aAns=0., aEps0=aEps;
     int N=50000*pow(2., (-log10(eps)-2)/2.);
     r_gauss(func,a,b,eps,&aEps,&ans,&aAns,&N,0,&err);
//printf("k=%d  aEps0=%E aEps=%E aAns=%E \n",k, aEps0, aEps, aAns);
     if(err) { if(err_code) *err_code=err; else 
               switch(err)
               { case 1: printf("gauss345: NaN in integrand\n"); break;
                 case 2: printf("gauss345: Too many points need for integration\n"); break;
                 case 3: printf("gauss345: Too deep recursion\n"); break;
               }
               return gauss(func,a,b,7);                                       
             }           
     if(aEps0-aEps < eps*aAns || k)   return ans; 
     aEps=aAns*eps;
  }
}

double gauss_arg( double (*func)(double,void*),void*par,double a,double b,  int n)
{
                
  double ans=0;
  if(n<1) n=1;
  if(n>7) { n=7; printf(" 7 is a miximum number of points for Gauss integration\n");} 
  switch(n)
  {  int i;
    case 1: ans=(b-a)*func((a+b)/2,par);  break;
    case 2:
      for(i=0;i<n;i++) ans+=F2[i]*func(a+ (b-a)*X2[i],par); break;
    case 3: 
      for(i=0;i<n;i++) ans+=F3[i]*func(a+ (b-a)*X3[i],par); break;
    case 4:
      for(i=0;i<n;i++) ans+=F4[i]*func(a+ (b-a)*X4[i],par); break;
    case 5:
      for(i=0;i<n;i++) ans+=F5[i]*func(a+ (b-a)*X5[i],par); break;
    case 6:
      for(i=0;i<n;i++) ans+=F6[i]*func(a+ (b-a)*X6[i],par); break;
    case 7:
      for(i=0;i<n;i++) ans+=F7[i]*func(a+ (b-a)*X7[i],par); break;      
  }
  return ans*(b-a);                       
 }


static double peterson21_stat(double *f, double a, double b, double *aerr)
{
// FormCalc/util/univariate/Patterson.F


// weights of the 10-point formula
  double  w10[]={
  0.066671344308688137593568809893332,
  0.149451349150580593145776339657697,
  0.219086362515982043995534934228163,
  0.269266719309996355091226921569469,
  0.295524224714752870173892994651338
                };

// weights of the 21-point formula
  double   w21[]={
  0.149445554002916905664936468389821,
  0.032558162307964727478818972459390,
  0.075039674810919952767043140916190,
  0.109387158802297641899210590325805,
  0.134709217311473325928054001771707,
  0.147739104901338491374841515972068,
  0.011694638867371874278064396062192,
  0.054755896574351996031381300244580,
  0.093125454583697605535065465083366,
  0.123491976262065851077958109831074,
  0.142775938577060080797094273138717 
                };

/*
  double f[21];
  double mi=0.5*(a+b);
  f[0]=F(mi);
  for(int i=0;i<10;i++)
  {  double d=0.5*(b-a)*x[i];
     f[1+i]=F(mi+d);
     f[11+i]=F(mi-d);
  } 
*/                                              
  double sum10=0, sum21=f[0]*w21[0];
 
  for(int i=1;i<=5; i++)  sum10+=w10[i-1]*(f[i]+f[i+10]);
  for(int i=1;i<=10;i++)  sum21+=w21[i]*(f[i]+f[i+10]);

  if(aerr)
  { 
     double h=sum21/2;
     double fluct=w21[0]*fabs(f[0]-h);
     for(int i=0;i<10;i++) fluct+=w21[i+1]*( fabs(f[i+1]-h) +fabs(f[i+11]-h));
     fluct*=fabs(b-a)/2;
     double err=fabs(sum21-sum10)*fabs(b-a)/2;
     if(fluct>1E-13) *aerr=err; else 
     {
        double er=pow(200*err/fluct,1.5);
        if(er>1) er=1;
        *aerr=fluct*er;
    }
  }
  
  return (b-a)*sum21/2;

} 

static double xPeterson[]=
{
  0.973906528517171720077964012084452,
  0.865063366688984510732096688423493,
  0.679409568299024406234327365114874,
  0.433395394129247190799265943165784,
  0.148874338981631210884826001129720,
  0.995657163025808080735527280689003,
  0.930157491355708226001207180059508,
  0.780817726586416897063717578345042,
  0.562757134668604683339000099272694,
  0.294392862701460198131126603103866
             };


double peterson21(double (*F)(double), double a, double b, double *aerr)
{
   double f[21];
   double mi=0.5*(a+b);
   f[0]=F(mi);
   for(int i=0;i<10;i++)
   {  double d=0.5*(b-a)*xPeterson[i];
      f[1+i]=F(mi+d);
      f[11+i]=F(mi-d);
   }      
   return  peterson21_stat(f, a, b, aerr);
}

double peterson21_arg(double (*F)(double,void*),void*par, double a, double b, double *aerr)
{
   double f[21];
   double mi=0.5*(a+b);
   f[0]=F(mi,par);
   for(int i=0;i<10;i++)
   {  double d=0.5*(b-a)*xPeterson[i];
      f[1+i]=F(mi+d,par);
      f[11+i]=F(mi-d,par);
   }      
   return  peterson21_stat(f, a, b, aerr);   
}

static void drawP(void*f, void*par, double x1, double x2,double ans,int nErr)
{  
   FILE*F=fopen("verifySimpson.tab","w");
   fprintf(F,"#title verifySimpson integral=%E \n", ans);
   fprintf(F,"#yName func{c}\n"); 
   fprintf(F,"#xName x\n");
   fprintf(F,"#xMin %E\n", x1); 
   fprintf(F,"#xMax %E\n", x2);
   fprintf(F,"#xDim  100\n");
   fprintf(F,"#xScale 0\n");
   fprintf(F,"#---   starting of data ---\n");
   fprintf(F,"#  func(x)\n");
   if(par)
   { double(*func)(double,void*)=f;
     for(int i=0;i<100;i++) fprintf(F," %E\n", func( x1+ (i+0.5)*(x2-x1)/100.,par));
   } else              
   { double(*func)(double)=f; 
     for(int i=0;i<100;i++) fprintf(F," %E\n", func( x1+ (i+0.5)*(x2-x1)/100.));
   }
   
   fclose(F);     
   char buff[1000];        
   sprintf(buff,"%s/bin/plot_view verifySimpson.tab",rootDir);
   system(buff);
}

void verifySimpson(void*f, void*par, double a, double b,double ans,int nErr)
{ int show=0; double x1=a, x2=b;
  
  while(show)  {drawP(f, par, x1, x2,ans,nErr);
     continue; } 
}

#ifdef TODELETE
static void r_simpson( double(*func)(double),double * f,double a,double b, 
double eps, double * ans,  double * absAns, double *dErr,  int depth, int depth1, int*nErr)
{
  double f1[9];
  int i;
  double s1,s2,s3,e_err;

  s1=(f[0]+4*f[4]+f[8])/6;  
  s2=(f[0]+4*f[2]+2*f[4]+4*f[6]+f[8])/12;
  s3=(f[0]+4*f[1]+2*f[2]+4*f[3]+2*f[4]+4*f[5]+2*f[6]+4*f[7]+f[8])/24;

/*
  double d1=fabs((s3-s2)*(b-a));
  double d2=fabs((s3-s1)*(b-a))/16;
  printf("a=%E b=%E  nErr=%d  d1=%.2E d2=%.2E absAns=%.2E ans=%.2E \n", a,b,*nErr,d1,d2,*absAns,*ans);
*/

  e_err=eps*fabs(s3);
  int ok=0;
  if( ( fabs(s3-s2) <= e_err && fabs(s3-s1) <= 16*e_err)) { *absAns+=(fabs(f[0])+4*fabs(f[1])+2*fabs(f[2])+4*fabs(f[3])+2*fabs(f[4])+4*fabs(f[5])+2*fabs(f[6])+4*fabs(f[7])+fabs(f[8]))/24*(b-a);   ok=1;}
   else
   if( fabs((s3-s2)*(b-a)) <= 0.1*eps*(*absAns) && fabs((s3-s1)*(b-a)) <= 1.6*eps*(*absAns))  { ok=1;  *absAns -= fabs((s3-s2)*(b-a))/eps;}
  if(ok) { *ans+=s3*(b-a); return; } 


  if(depth >= 8)
  {
    int c=0, inc;
    inc=f[0]<f[1];   
    for(i=1;i<7;i++) { if(inc){ if(f[i]>f[i+1]) {inc=0;c++;}} else if(f[i]<f[i+1]){ inc=1; c++;}}
    if(c>3) 
    {  double ff=0.5*(f[0]+f[8]); for(i=1;i<8;i++)ff+=f[i]; ff/=8;
       double ff2=0.5*(f[0]*f[0]+f[8]*f[8]); for(i=1;i<8;i++)ff2+=f[i]*f[i]; ff2/=8;
       double df=sqrt(ff2-ff*ff)/pow(2,0.5*depth);
//       printf("ff=%e df=%e\n", ff,df);       
printf("lost of precision [%e %e]\n", a,b);
       if( df<0.01*fabs(ff)) { *nErr=*nErr|4;   *ans+= ff*(b-a);  return;}       
    }   
  }

  
  if(depth>depth1) { /**nErr=*nErr|2;*/ *ans+=s3*(b-a);  *dErr+=fabs((s3-s2)*(b-a));  return; }
   
  for(i=0;i<5;i++) f1[2*i]=f[4+i];
  for(i=8;i>0;i-=2)f[i]=f[i/2];
   
  for(i=1;i<8;i+=2) { f[i]=(*func)(a+i*(b-a)/16);         if(!isfinite(f[i]))  {f[i]=0;   *nErr=*nErr|1;}  }   //   *N-=4;
  for(i=1;i<8;i+=2) { f1[i]=(*func)((a+b)/2+i*(b-a)/16);  if(!isfinite(f1[i])) {f1[i]=0;  *nErr=*nErr|1;}  }   //   *N-=4;
    

  double as= -f[0]+4*f[1]-6*f[2]+4*f[3]-2*f[4]+4*f[5]-6*f[6]+4*f[7]-f[8];
  double as1= -f1[0]+4*f1[1]-6*f1[2]+4*f1[3]-2*f1[4]+4*f1[5]-6*f1[6]+4*f1[7]-f1[8];
   s3=(f[0]+4*f[1]+2*f[2]+4*f[3]+2*f[4]+4*f[5]+2*f[6]+4*f[7]+f[8])/24;
  double  s31=(f1[0]+4*f1[1]+2*f1[2]+4*f1[3]+2*f1[4]+4*f1[5]+2*f1[6]+4*f1[7]+f1[8])/24;
  
  if(as*s31<as1*s3)
  {
    r_simpson_arg(func,NULL,a,     (a+b)/2,eps,  f, ans,absAns,dErr, depth+1,depth1,nErr);
    r_simpson_arg(func,NULL,(a+b)/2,b,     eps, f1, ans,absAns,dErr, depth+1,depth1,nErr);
  } else 
  {
    r_simpson_arg(func,NULL,(a+b)/2,b,      eps, f1, ans,absAns,dErr,depth+1,depth1,nErr);
    r_simpson_arg(func,NULL, a,     (a+b)/2,eps, f,  ans,absAns,dErr,depth+1,depth1,nErr);
  }

}
#endif

static void r_simpson_arg( void*F, void*par,  double a,double b,  
                           double eps, double * f,  double * ans, double * absAns,double*dErr, int depth,int depth1,int*nErr)
{
  double f1[9];
  int i;
  double s1,s2,s3,e_err;
  
  s1=(f[0]+4*f[4]+f[8])/6;
  s2=(f[0]       +4*f[2]       +2*f[4]       +4*f[6]       +f[8])/12;
  s3=(f[0]+4*f[1]+2*f[2]+4*f[3]+2*f[4]+4*f[5]+2*f[6]+4*f[7]+f[8])/24;
  
  
  e_err=eps*fabs(s3);
  
//  double d1=fabs((s3-s2)*(b-a));
//  double d2=fabs((s3-s1)*(b-a))/16;
//  printf("a=%E b=%E  nErr=%d  d1=%.2E d2=%.2E absAns=%.2E ans=%.2E \n", a,b,*nErr,d1,d2,*absAns,*ans);
  
  int ok=0;
       if( ( fabs(s3-s2) <= e_err && fabs(s3-s1) <= 16*e_err)) { *absAns+=(fabs(f[0])+4*fabs(f[1])+2*fabs(f[2])+4*fabs(f[3])+2*fabs(f[4])+4*fabs(f[5])+2*fabs(f[6])+4*fabs(f[7])+fabs(f[8]))/24*(b-a);   ok=1; }
  else if( fabs((s3-s2)*(b-a)) <= 0.1*eps*(*absAns) && fabs((s3-s1)*(b-a)) <= 1.6*eps*(*absAns))  { ok=1;  *absAns -= fabs((s3-s2)*(b-a))/eps;}


  if(ok) { *ans+=s3*(b-a); return; } 
  
  if(depth >= 8)
  {
    int c=0, inc;
    inc=f[0]<f[1];   
    for(i=1;i<7;i++) { if(inc){ if(f[i]>f[i+1]) {inc=0;c++;}} else if(f[i]<f[i+1]){ inc=1; c++;}}
    if(c>=3) 
    {  double ff=0.5*(f[0]+f[8]); for(i=1;i<8;i++)ff+=f[i]; ff/=8;    
       double ff2=0.5*(f[0]*f[0]+f[8]*f[8]); for(i=1;i<8;i++)ff2+=f[i]*f[i]; ff2/=8;
       double df=sqrt(ff2-ff*ff)/pow(2,0.5*depth);
       if( df>0.01*fabs(ff)) 
       { *nErr=*nErr|4; *ans+= ff*(b-a); return;}
    }   
  }
  
  if(depth>depth1) { *ans+=s3*(b-a);  *dErr+=fabs((s3-s2)*(b-a));  return; }
  
   
  for(i=0;i<5;i++) f1[2*i]=f[4+i];
  for(i=8;i>0;i-=2)f[i]=f[i/2];

  if(par)
  { double(*func)(double,void*par)=F;
    for(i=1;i<8;i+=2) { f[i]=(*func)(a+i*(b-a)/16,par);         if(!isfinite(f[i])) { f[i]=0;  *nErr=*nErr|1;}    }  //   *N-=4;
    for(i=1;i<8;i+=2) { f1[i]=(*func)((a+b)/2+i*(b-a)/16,par);  if(!isfinite(f1[i])){ f1[i]=0; *nErr=*nErr|1;}    }  //   *N-=4;
  } else 
  { double(*func)(double)=F;
    for(i=1;i<8;i+=2) { f[i]=(*func)(a+i*(b-a)/16 );            if(!isfinite(f[i])) { f[i]=0;  *nErr=*nErr|1;}    }  //   *N-=4;
    for(i=1;i<8;i+=2) { f1[i]=(*func)((a+b)/2+i*(b-a)/16);      if(!isfinite(f1[i])){ f1[i]=0; *nErr=*nErr|1;}    }  //   *N-=4;
  }

  double as= -f[0]+4*f[1]-6*f[2]+4*f[3]-2*f[4]+4*f[5]-6*f[6]+4*f[7]-f[8];
  double as1= -f1[0]+4*f1[1]-6*f1[2]+4*f1[3]-2*f1[4]+4*f1[5]-6*f1[6]+4*f1[7]-f1[8];
  if(fabs(as)<fabs(as1))
  {
    r_simpson_arg(F,par,   a,     (a+b)/2, eps, f,  ans,absAns,dErr,depth+1,depth1,nErr);
    r_simpson_arg(F,par,  (a+b)/2,    b,   eps, f1, ans,absAns,dErr,depth+1,depth1,nErr);
  } else 
  {  r_simpson_arg(F,par, (a+b)/2,b,       eps, f1, ans,absAns,dErr,depth+1,depth1,nErr);
     r_simpson_arg(F,par,   a,     (a+b)/2,eps, f,  ans,absAns,dErr,depth+1,depth1,nErr);  
  }  
}


double simpson( double (*func)(double),double a,double b, double  eps, int *err)
{
  if(a==b) return 0;

  int nErr=0;
  int depth1= 15-log(eps);
   
  double ans=0,absAns=0,dErr=0;
  double f[9];  
  for(int i=0;i<9;i++) { f[i]=(*func)(a+i*(b-a)/8); if(!isfinite(f[i])){ f[i]=0; nErr=nErr|1;}  }

  r_simpson_arg(func,NULL,a,b,eps, f,&ans, &absAns,&dErr, 0,depth1, &nErr);
  
  if(dErr> 0.1*eps*absAns) { nErr=nErr|2;}
  
  if(err) *err=nErr;
  if(nErr)
  { 
    verifySimpson(func,NULL,a,b,ans,nErr); 
    if(!err) 
    {  printf("simpson warnings:");  
       if(nErr & 1) printf("NaN in integrand; ");  
       if(nErr & 2) printf("Too deep recursion; ");             
       if(nErr & 4) printf("Lost of precision."); 
       printf("\n");
    }  
  }
  return ans;  
}



double simpson_arg( double (*func)(double,void*),void*par,double a,double b, double  eps, int *err)
{

  if(a==b) return 0;

  int nErr=0;	
  int depth1=  25-log(eps);
  
  double ans=0., absAns=0.,dErr=0; 
  double f[9];
  for(int i=0;i<9;i++) { f[i]=(*func)(a+i*(b-a)/8,par); if(!isfinite(f[i])) {f[i]=0; nErr=1;} }
   
  r_simpson_arg(func,par,a,b,eps,f,&ans,&absAns,&dErr,0,depth1,&nErr);
  if(dErr> 0.1*eps*absAns) { nErr=nErr|2;}
   
  if(err) *err=nErr;
  if(nErr)
  {
     verifySimpson(func,par,a,b,ans,nErr);
     if(!err)
     { printf("simpson warnings:");  
       if(nErr & 1) printf("NaN in integrand; ");  
       if(nErr & 2) printf("Too deep recursion; ");             
       if(nErr & 4) printf("Lost of precision."); 
       printf("\n");
     }
  }     
  return ans;    
}


static void r_gauss_arg( double(*func)(double,void*),void*par, double a,double b, 
double eps, double * aEps, double * ans, double * aAns,int* N, int depth, int*err)
{
  int n;

  double s1,s2,s2a,s3,s3a,e_err,d=b-a;

  if(*N<0)      { *err=2;  return; }
  if(depth>50)  { *err=3;  return; }

  for(n=0,s1=0;n<3;n++) s1+=F3[n]*func(a+ d*X3[n],par); s1*=d; *N-=3;
  for(n=0,s2=0,s2a=0;n<4;n++) {double f=F4[n]*func(a+ d*X4[n],par); s2+=f; s2a+=fabs(f);} 
  s2*=d; s2a*=fabs(d);*N-=4;
  
  if(!isfinite(s1) || !isfinite(s2)) { *err=1; return;}
 
  e_err=eps*fabs(s2);
 
  if( fabs(s1-s2) <= 30*e_err)
  { 
    for(n=0,s3=0,s3a=0;n<5;n++) {double f=F5[n]*func(a+ d*X5[n],par); s3+=f; s3a+=fabs(f);} 
    if(!isfinite(s3)) { *err=1; return;} 
    s3*=d; s3a*=d; *N-=5;
    if(fabs(s3-s2) <= e_err) 
    { *ans+=s3;
      *aAns+=s3a;
      return;
    }
  }    

  if(fabs(s1-s2) <= 0.1*(*aEps)) 
  {  *ans+=s2;
     *aAns+=s2a;   
     *aEps -= fabs(s1-s2);
     return;
  }
  
  r_gauss_arg(func,par,a,(a+b)/2,eps,aEps,ans,aAns,N,depth+1,err);
  if(*err)  return;
  r_gauss_arg(func,par,(a+b)/2,b,eps,aEps,ans,aAns,N,depth+1,err);
}   

double gauss345_arg( double (*func)(double,void*),void*par,double a,double b, double eps,int * err_code)
{
  double d,s1,s2,s2a,s3,s3a,aEps; /* absolute error  */
  int k,n,err=0;	

  if(a==b) return 0;
  d=b-a;
  for(n=0,s1=0;n<3;n++) s1+=F3[n]*func(a+ d*X3[n],par); s1*=d;
  for(n=0,s2=0,s2a=0;n<4;n++) {double f=F4[n]*func(a+ d*X4[n],par); s2+=f; s2a+=fabs(f);}
       s2*=d; s2a*=fabs(d);
       
  if(err_code) *err_code=0;
  
  

  if(!isfinite(s1) || !isfinite(s2)  ) 
  { if(err_code) *err_code=1; else printf("gauss345_arg: NaN in integrand\n"); 
    return 0;
  }
       
  aEps= s2a*eps;
  if( fabs(s1-s2) <= 30*aEps)
  { 
    for(n=0,s3=0,s3a=0;n<5;n++) {double f=F5[n]*func(a+ d*X5[n],par); s3+=f; s3a+=fabs(f);} 
    if(!isfinite(s3)) 
    {  if(err_code) *err_code=1; 
       else printf("gauss345_arg: NaN in integrand\n"); 
       return 0;
    } 
    s3*=d; s3a*=fabs(d);
    aEps=eps*s3a;
    if(fabs(s3-s2) < aEps)  return s3; 
  }    

  aEps/=2;
  eps=eps/2;

  for(k=0;;k++)
  {  double ans=0., aAns=0., aEps0=aEps;
     int N=50000*pow(2., (-log10(eps)-2)/2.);
     r_gauss_arg(func,par,a,b,eps,&aEps,&ans,&aAns,&N,0,&err);
//printf("k=%d  aEps0=%E aEps=%E aAns=%E \n",k, aEps0, aEps, aAns);
     if(err) 
     {   if(err_code) *err_code=err;  else 
         switch(err)
         { case 1: printf("gauss345_arg: NaN in integrand\n");  break;
           case 2: printf("gauss345_arg: Too many points need for integration\n"); break;
           case 3: printf("gauss345_arg: Too deep recursion\n"); break;
         }
         return gauss_arg(func,par,a,b,7);
     }     
     if(aEps0-aEps < 1.5*eps*aAns || k)    return ans;
     aEps=aAns*eps;
  } 
}

