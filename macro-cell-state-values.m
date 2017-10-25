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
soln
]

pythonTuples[states_]:=Module[{keys,key,i,j},
keys={};
For[i=1,i<=Length[states],i++,
key="(";
For[j=1,j<=Length[states[[i]]],j++,
If[j<Length[states[[i]]],
key=key<>ToString[states[[i,j]]]<>",",
key=key<>ToString[states[[i,j]]]
];
];
key=key<>")";
AppendTo[keys,key];
];
keys
]

values[arrRates_, servRates_, truncation_, idlePower_, busyPower_, export_:0]:=Module[{loads,n,perfCoeffs,energyCoeffs,states,keys,values,filename},
n=truncation;
perfCoeffs = solveCoefficients[arrRates,servRates,idlePower,busyPower,0];
energyCoeffs = solveCoefficients[arrRates,servRates,idlePower,busyPower,1];
states = Tuples[Range[0,n],Length[arrRates]];
If[export==0,
values = (#->{(valueFunction[states[[#]]]/.perfCoeffs), energyValue[states[[#]], energyCoeffs]})&/@Range[Length[states]],

keys=pythonTuples[states];
values = (keys[[#]]->{valueFunction[states[[#]]]/.perfCoeffs, energyValue[states[[#]], energyCoeffs]})&/@Range[Length[states]];

loads=Round[N[arrRates/servRates], 0.001];
loads = StringJoin["-"<>ToString[#]&/@loads];
filename = "/Users/misikir/Google\ Drive/EIT-Project/code/json/state_values_load"<>loads<>".json";
Export[filename,values]
];
{values, perfCoeffs}
]

(*arrRates=ToExpression[$ScriptCommandLine[[2]]];
servRates=ToExpression[$ScriptCommandLine[[3]]]*)



params = $ScriptCommandLine

lastThree = Take[params, -3];

truncation = ToExpression[lastThree[[1]]];
idlePower  = ToExpression[lastThree[[2]]];
busyPower  = ToExpression[lastThree[[3]]];

rates = Take[params, {2,-4}];
rates = Partition[rates, Length[rates]/2];
arrRates = ToExpression[rates[[1]]];
servRates = ToExpression[rates[[2]]];


values[arrRates, servRates, truncation, idlePower, busyPower, 1];

Exit[];
