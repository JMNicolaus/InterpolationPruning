function [xy,out] = GenData4Cat(n,a,b)
% this function samples [x y] for a <= x <= b and f_min <= y <= f_max 
% outputs a 4x1 vector with 1 in the i-th component, if [x,y] belongs to
% category i    

  f1 = @(x) x.^2-1;
  f2 = @(x) -1*x.^2+1;
  
  % draw sample points
  x = (b-a)*rand(n,1)+a;
  f1x = f1(x);
  f2x = f2(x);
  fmax = max(max(f1x,f2x));
  fmin = min(min(f1x,f2x));
  y = (fmax-fmin)*rand(n,1)+fmin;
  xy = [x';y'];

  % categorize
  out = zeros(4,n);
  out(1,all([y>=f1x y>=f2x],2) )=1;
  out([2 3 4],all([y>=f1x y>=f2x],2) )=0;

  out(2,all([y<=f1x y<=f2x],2))=1;
  out([1 3 4],all([y<=f1x y<=f2x],2))=0;

  out(3,all([y<f1x y>f2x],2))=1;
  out([1 2 4],all([y<f1x y>f2x],2))=0;

  out(4,all([y>f1x y<f2x],2))=1;
  out([1 2 3],all([y>f1x y<f2x],2))=0;
end

