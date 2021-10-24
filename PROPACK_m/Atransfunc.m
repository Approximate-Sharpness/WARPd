function y=Atransfunc(x)
% y=Atransfunc(x)
% Testfunction returning the transpose of a linear operator applied to x.
% Used for testing lansvd.
%
% y = A'*x


global A MxV
y = A'*x;
MxV = MxV + 1; 
