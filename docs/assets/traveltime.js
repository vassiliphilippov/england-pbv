const DRIVE_PARAMS={"coeff":3.0014999999999996,"exponent":0.809};
function driveMinutes(crowKm){
  return DRIVE_PARAMS.coeff*Math.pow(Math.max(crowKm,0),DRIVE_PARAMS.exponent);
}
function haversineKm(la1,lo1,la2,lo2){
  const r=Math.PI/180, p1=la1*r, p2=la2*r, dp=(la2-la1)*r, dl=(lo2-lo1)*r;
  const a=Math.sin(dp/2)**2+Math.cos(p1)*Math.cos(p2)*Math.sin(dl/2)**2;
  return 2*6371*Math.asin(Math.sqrt(a));
}
