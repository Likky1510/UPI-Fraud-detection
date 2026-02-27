const transactions = [
{id:"TXN1",amount:5000,fraud:false,risk:10},
{id:"TXN2",amount:20000,fraud:true,risk:85},
{id:"TXN3",amount:1500,fraud:false,risk:5},
{id:"TXN4",amount:45000,fraud:true,risk:92}
];

let total=transactions.length;
let fraud=transactions.filter(t=>t.fraud).length;
let safe=total-fraud;

document.getElementById("total").innerText=total;
document.getElementById("fraud").innerText=fraud;
document.getElementById("safe").innerText=safe;

const table=document.getElementById("tableData");

transactions.forEach(t=>{
table.innerHTML+=`
<tr>
<td>${t.id}</td>
<td>₹${t.amount}</td>
<td style="color:${t.fraud?'#ff4d4d':'#00ff99'}">
${t.fraud?'Fraud':'Safe'}
</td>
<td>${t.risk}%</td>
</tr>`;
});

new Chart(document.getElementById("chart"),{
type:'bar',
data:{
labels:["Safe","Fraud"],
datasets:[{
label:"Transactions",
data:[safe,fraud]
}]
}
});