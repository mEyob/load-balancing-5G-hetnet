#!/usr/bin/env WolframScript -script

valueFunction[xvec_]:=Sum[Subscript[a, i,i] (xvec[[i]]) ^2,{i,1,Length[xvec]}]+ 2Sum[Sum[Subscript[a, i,j] xvec[[i]] xvec[[j]],{j,i+1,Length[xvec]}],{i,1,Length[xvec]-1}]+Sum[Subscript[a, i] xvec[[i]] ,{i,1,Length[xvec]}]

avgCostRate[arrRates_,servRates_]:=Sum[arrRates[[i]]/servRates[[i]],{i,1,Length[arrRates]}]/(1-Sum[arrRates[[i]]/servRates[[i]],{i,1,Length[arrRates]}])

howardEquation[xvec_,arrRates_,servRates_]:=Sum[ xvec[[i]] ,{i,1,Length[xvec]}]*(Sum[ xvec[[i]],{i,1,Length[xvec]}]-avgCostRate[arrRates,servRates] +Sum[arrRates[[i]](valueFunction[xvec+UnitVector[Length[xvec],i]]-valueFunction[xvec]) ,{i,1,Length[xvec]}])+Sum[servRates[[i]]xvec[[i]](valueFunction[xvec-UnitVector[Length[xvec],i]]-valueFunction[xvec]) ,{i,1,Length[xvec]}]


power[xvec_, idlePower_, busyPower_]:=Piecewise[{{idlePower, Total[xvec]==0},{busyPower, Total[xvec]!=0}}]
avgEnergyCostRate[arrRates_,servRates_, idlePower_, busyPower_]:=Total[arrRates/servRates]*busyPower + (1-Total[arrRates/servRates])*idlePower
energyHowardEquation[xvec_,arrRates_,servRates_, idlePower_, busyPower_]:=Module[{},
Sum[ xvec[[i]] ,{i,1,Length[xvec]}]*(power[xvec,idlePower,busyPower]-avgEnergyCostRate[arrRates,servRates, idlePower, busyPower] +Sum[arrRates[[i]](valueFunction[xvec+UnitVector[Length[xvec],i]]-valueFunction[xvec]) ,{i,1,Length[xvec]}])+Sum[servRates[[i]]xvec[[i]](valueFunction[xvec-UnitVector[Length[xvec],i]]-valueFunction[xvec]) ,{i,1,Length[xvec]}]
]
energyValue[state_,coeffs_]:=Module[{xvec,x,coeff,value},
coeff = If[Total[state] == 0, coeffs[[1]], coeffs[[2]]];
value = valueFunction[state]/.coeff;
value
]
solveCoefficients[arrRates_,servRates_, idlePower_, busyPower_, energy_:0]:=Module[{xvec,x,poly,eqn,variables,linear,soln,i},
xvec=Table[Subscript[x, i],{i,1,Length[arrRates]}];
poly = (#*xvec&/@xvec);
AppendTo[poly,xvec];
poly = DeleteDuplicates[Flatten[poly]];
If[energy==0,
eqn=Collect[howardEquation[xvec,arrRates,servRates]//Expand,poly],
eqn=Collect[energyHowardEquation[xvec,arrRates,servRates, idlePower, busyPower]//Expand,poly]
];
variables = Flatten[Table[Subscript[a, i,j],{i,1,Length[arrRates]},{j,i,Length[arrRates]}]];
linear = Table[ Subscript[a, i],{i,1,Length[arrRates]}];
variables=Flatten[AppendTo[variables,linear]];

soln = Solve[(Coefficient[eqn,#]==0)&/@poly, variables];
soln=soln[[1]];
If[energy!=0, soln={soln/.{Total[xvec]->0}, soln/.{Total[xvec]->1}},
soln=<|soln|>;
For[i=1,i<=Length[linear],i++,soln[Subscript[a, i]]=soln[Subscript[a, i,i]]];
soln=Normal[soln]];
soln=<|soln|>;

soln

]



export[arrRates_, servRates_, idlePower_, busyPower_]:=Module[{perfCoeffs, energyCoeffs, coeffs, cLin, cQuad, loads, fileName},
(***PERFORMANCE PART ******)
perfCoeffs = solveCoefficients[arrRates,servRates,idlePower,busyPower,0];
cLin = Table[ToString[i-1]->perfCoeffs[Subscript[a, i]], {i,1,Length[arrRates]}];
cQuad = Table["("<>ToString[i-1]<>","<>ToString[j-1]<>")"->perfCoeffs[Subscript[a, i,j]], {i,1,Length[arrRates]},{j,i,Length[arrRates]}];
perfCoeffs = Flatten[AppendTo[cQuad, cLin]];

(***ENERGY PART ******)

energyCoeffs = solveCoefficients[arrRates,servRates,idlePower,busyPower,1];
cLin = Table[ToString[i-1]->energyCoeffs[Subscript[a, i]], {i,1,Length[arrRates]}];
cQuad = Table["("<>ToString[i-1]<>","<>ToString[j-1]<>")"->energyCoeffs[Subscript[a, i,j]], {i,1,Length[arrRates]},{j,i,Length[arrRates]}];
energyCoeffs = Flatten[AppendTo[cQuad, cLin]];

(***Combining both ****)
coeffs = {"perfCoeffs"->perfCoeffs, "energyCoeffs"->energyCoeffs};
loads=Round[N[arrRates/servRates],0.001];
loads = StringJoin["-"<>ToString[#]&/@loads];
dir = Directory[];

fileName = dir<>"/json/"<>"coeffs_load"<>loads<>".json";
Export[fileName,coeffs];

]


(*export[{0.1,0.9,0.5},{1,2,2},700,1000]*)



params = $ScriptCommandLine

lastThree = Take[params, -2];


idlePower  = ToExpression[lastThree[[1]]];
busyPower  = ToExpression[lastThree[[2]]];

rates = Take[params, {2,-3}];
rates = Partition[rates, Length[rates]/2];
arrRates = ToExpression[rates[[1]]];
servRates = ToExpression[rates[[2]]];


export[arrRates, servRates, idlePower, busyPower];

Exit[];
