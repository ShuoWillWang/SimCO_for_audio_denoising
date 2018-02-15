function [D]  = MOD(Y,param)

itN = param.itN;
D = param.initialDictionary;
errglobal = param.errorGoal;

for itn = 1:itN
    
    X = OMPerr(D,Y,errglobal);
    
    D = M_DictUpdate02 (Y,X);

    f=sum(sum((Y-D*X).^2));
    fprintf('\n ** final cost function value is %3.3e ** \n\n',...
            f);
     
end


end
