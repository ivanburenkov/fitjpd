//Conjugated modes
var cthList = document.getElementById("thermal-sources");
function addConjugatedThermalMode() {
  cthn += 1;
  var countCTh = cthList.childElementCount;
  var valsCTh = [];
  var j = 0;
  for (i = 0; i < countCTh; i++) {
    valsCTh[i] = document.getElementById("cth" + i).value;
  }
  cthList.innerHTML = "";
  for (i = 0; i < countCTh + 1; i++) {
    if (!valsCTh[i]) valsCTh[i] = 0.5;
    cthList.innerHTML +=
      "<li id='cthli" +
      i +
      "'><div class='input-group'><input id='cth" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 class='form-inline' value='" +
      valsCTh[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delConjugatedThermalMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  feather.replace();
}

function delConjugatedThermalMode(liId) {
  var countCTh = cthList.childElementCount;
  var valsCTh = [];
  var j = 0;
  for (i = 0; i < countCTh; i++) {
    if (i != liId) {
      valsCTh[j] = document.getElementById("cth" + i).value;
      j++;
    }
  }
  cthList.innerHTML = "";
  for (i = 0; i < countCTh - 1; i++) {
    document.getElementById("thermal-sources").innerHTML +=
      "<li id='cthli" +
      i +
      "'><div class='input-group'><input id='cth" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 class='form-inline' value='" +
      valsCTh[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delConjugatedThermalMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  cthn -= 1;
  feather.replace();
  //runCcode();
}

var cpoiList = document.getElementById("poisson-sources");
function addConjugatedPoissonMode() {
  cpoin += 1;
  var countCPoi = cpoiList.childElementCount;
  var valsCPoi = [];
  var j = 0;
  for (i = 0; i < countCPoi; i++) {
    valsCPoi[i] = document.getElementById("cpoi" + i).value;
  }
  cpoiList.innerHTML = "";
  for (i = 0; i < countCPoi + 1; i++) {
    if (!valsCPoi[i]) valsCPoi[i] = 0.5;
    cpoiList.innerHTML +=
      "<li id='cpoili" +
      i +
      "'><div class='input-group'><input id='cpoi" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 class='form-inline' value='" +
      valsCPoi[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delConjugatedPoissonMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  feather.replace();
}

function delConjugatedPoissonMode(liId) {
  var countCPoi = cpoiList.childElementCount;
  var valsCPoi = [];
  var j = 0;
  for (i = 0; i < countCPoi; i++) {
    if (i != liId) {
      valsCPoi[j] = document.getElementById("cpoi" + i).value;
      j++;
    }
  }
  cpoiList.innerHTML = "";
  for (i = 0; i < countCPoi - 1; i++) {
    document.getElementById("poisson-sources").innerHTML +=
      "<li id='cpoili" +
      i +
      "'><div class='input-group'><input id='cpoi" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 class='form-inline' value='" +
      valsCPoi[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delConjugatedPoissonMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  cpoin -= 1;
  feather.replace();
  //runCcode();
}

var cspList = document.getElementById("sp-sources");
function addConjugatedSPMode() {
  cspn += 1;
  var countCSP = cspList.childElementCount;
  var valsCSP = [];
  var j = 0;
  for (i = 0; i < countCSP; i++) {
    valsCSP[i] = document.getElementById("csp" + i).value;
  }
  cspList.innerHTML = "";
  for (i = 0; i < countCSP + 1; i++) {
    if (!valsCSP[i]) valsCSP[i] = 0.5;
    cspList.innerHTML +=
      "<li id='cspli" +
      i +
      "'><div class='input-group'><input id='csp" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 max=1 step=0.1 class='form-inline' value='" +
      valsCSP[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delConjugatedSPMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  feather.replace();
}

function delConjugatedSPMode(liId) {
  var countCSP = cspList.childElementCount;
  var valsCSP = [];
  var j = 0;
  for (i = 0; i < countCSP; i++) {
    if (i != liId) {
      valsCSP[j] = document.getElementById("csp" + i).value;
      j++;
    }
  }
  cspList.innerHTML = "";
  for (i = 0; i < countCSP - 1; i++) {
    document.getElementById("sp-sources").innerHTML +=
      "<li id='cspli" +
      i +
      "'><div class='input-group'><input id='csp" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 max=1 step=0.1  class='form-inline' value='" +
      valsCSP[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delConjugatedSPMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  cspn -= 1;
  feather.replace();
  //runCcode();
}
//Signal modes
var sthList = document.getElementById("sthermal-sources");
function addSignalThermalMode() {
  sthn += 1;
  var countSTh = sthList.childElementCount;
  var valsSTh = [];
  var j = 0;
  for (i = 0; i < countSTh; i++) {
    valsSTh[i] = document.getElementById("sth" + i).value;
  }
  sthList.innerHTML = "";
  for (i = 0; i < countSTh + 1; i++) {
    if (!valsSTh[i]) valsSTh[i] = 0.5;
    sthList.innerHTML +=
      "<li id='sthli" +
      i +
      "'><div class='input-group'><input id='sth" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 class='form-inline' value='" +
      valsSTh[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delSignalThermalMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  feather.replace();
}

function delSignalThermalMode(liId) {
  var countSTh = sthList.childElementCount;
  var valsSTh = [];
  var j = 0;
  for (i = 0; i < countSTh; i++) {
    if (i != liId) {
      valsSTh[j] = document.getElementById("sth" + i).value;
      j++;
    }
  }
  sthList.innerHTML = "";
  for (i = 0; i < countSTh - 1; i++) {
    document.getElementById("sthermal-sources").innerHTML +=
      "<li id='sthli" +
      i +
      "'><div class='input-group'><input id='sth" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 class='form-inline' value='" +
      valsSTh[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delSignalThermalMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  sthn -= 1;
  feather.replace();
  //runCcode();
}

var spoiList = document.getElementById("spoisson-sources");
function addSignalPoissonMode() {
  spoin += 1;
  var countSPoi = spoiList.childElementCount;
  var valsSPoi = [];
  var j = 0;
  for (i = 0; i < countSPoi; i++) {
    valsSPoi[i] = document.getElementById("spoi" + i).value;
  }
  spoiList.innerHTML = "";
  for (i = 0; i < countSPoi + 1; i++) {
    if (!valsSPoi[i]) valsSPoi[i] = 0.5;
    spoiList.innerHTML +=
      "<li id='spoili" +
      i +
      "'><div class='input-group'><input id='spoi" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 class='form-inline' value='" +
      valsSPoi[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delSignalPoissonMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  feather.replace();
}

function delSignalPoissonMode(liId) {
  var countSPoi = spoiList.childElementCount;
  var valsSPoi = [];
  var j = 0;
  for (i = 0; i < countSPoi; i++) {
    if (i != liId) {
      valsSPoi[j] = document.getElementById("spoi" + i).value;
      j++;
    }
  }
  spoiList.innerHTML = "";
  for (i = 0; i < countSPoi - 1; i++) {
    document.getElementById("spoisson-sources").innerHTML +=
      "<li id='spoili" +
      i +
      "'><div class='input-group'><input id='spoi" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 class='form-inline' value='" +
      valsSPoi[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delSignalPoissonMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  spoin -= 1;
  feather.replace();
  //runCcode();
}

var sspList = document.getElementById("ssp-sources");
function addSignalSPMode() {
  sspn += 1;
  var countSSP = sspList.childElementCount;
  var valsSSP = [];
  var j = 0;
  for (i = 0; i < countSSP; i++) {
    valsSSP[i] = document.getElementById("ssp" + i).value;
  }
  sspList.innerHTML = "";
  for (i = 0; i < countSSP + 1; i++) {
    if (!valsSSP[i]) valsSSP[i] = 0.5;
    sspList.innerHTML +=
      "<li id='sspli" +
      i +
      "'><div class='input-group'><input id='ssp" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 max=1 step=0.1 class='form-inline' value='" +
      valsSSP[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delSignalSPMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  feather.replace();
}

function delSignalSPMode(liId) {
  var countSSP = sspList.childElementCount;
  var valsSSP = [];
  var j = 0;
  for (i = 0; i < countSSP; i++) {
    if (i != liId) {
      valsSSP[j] = document.getElementById("ssp" + i).value;
      j++;
    }
  }
  sspList.innerHTML = "";
  for (i = 0; i < countSSP - 1; i++) {
    document.getElementById("ssp-sources").innerHTML +=
      "<li id='sspli" +
      i +
      "'><div class='input-group'><input id='ssp" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 max=1 step=0.1  class='form-inline' value='" +
      valsSSP[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delSignalSPMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  sspn -= 1;
  feather.replace();
  //runCcode();
}
//Idler modes
//Idler modes
var ithList = document.getElementById("ithermal-sources");
function addIdlerThermalMode() {
  ithn += 1;
  var countITh = ithList.childElementCount;
  var valsITh = [];
  var j = 0;
  for (i = 0; i < countITh; i++) {
    valsITh[i] = document.getElementById("ith" + i).value;
  }
  ithList.innerHTML = "";
  for (i = 0; i < countITh + 1; i++) {
    if (!valsITh[i]) valsITh[i] = 0.5;
    ithList.innerHTML +=
      "<li id='ithli" +
      i +
      "'><div class='input-group'><input id='ith" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 class='form-inline' value='" +
      valsITh[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delIdlerThermalMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  feather.replace();
}

function delIdlerThermalMode(liId) {
  var countITh = ithList.childElementCount;
  var valsITh = [];
  var j = 0;
  for (i = 0; i < countITh; i++) {
    if (i != liId) {
      valsITh[j] = document.getElementById("ith" + i).value;
      j++;
    }
  }
  ithList.innerHTML = "";
  for (i = 0; i < countITh - 1; i++) {
    document.getElementById("ithermal-sources").innerHTML +=
      "<li id='ithli" +
      i +
      "'><div class='input-group'><input id='ith" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 class='form-inline' value='" +
      valsITh[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delIdlerThermalMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  ithn -= 1;
  feather.replace();
  //runCcode();
}

var ipoiList = document.getElementById("ipoisson-sources");
function addIdlerPoissonMode() {
  ipoin += 1;
  var countIPoi = ipoiList.childElementCount;
  var valsIPoi = [];
  var j = 0;
  for (i = 0; i < countIPoi; i++) {
    valsIPoi[i] = document.getElementById("ipoi" + i).value;
  }
  ipoiList.innerHTML = "";
  for (i = 0; i < countIPoi + 1; i++) {
    if (!valsIPoi[i]) valsIPoi[i] = 0.5;
    ipoiList.innerHTML +=
      "<li id='ipoili" +
      i +
      "'><div class='input-group'><input id='ipoi" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 class='form-inline' value='" +
      valsIPoi[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delIdlerPoissonMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  feather.replace();
}

function delIdlerPoissonMode(liId) {
  var countIPoi = ipoiList.childElementCount;
  var valsIPoi = [];
  var j = 0;
  for (i = 0; i < countIPoi; i++) {
    if (i != liId) {
      valsIPoi[j] = document.getElementById("ipoi" + i).value;
      j++;
    }
  }
  ipoiList.innerHTML = "";
  for (i = 0; i < countIPoi - 1; i++) {
    document.getElementById("ipoisson-sources").innerHTML +=
      "<li id='ipoili" +
      i +
      "'><div class='input-group'><input id='ipoi" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 class='form-inline' value='" +
      valsIPoi[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delIdlerPoissonMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  ipoin -= 1;
  feather.replace();
  //runCcode();
}

var ispList = document.getElementById("isp-sources");
function addIdlerSPMode() {
  ispn += 1;
  var countISP = ispList.childElementCount;
  var valsISP = [];
  var j = 0;
  for (i = 0; i < countISP; i++) {
    valsISP[i] = document.getElementById("isp" + i).value;
  }
  ispList.innerHTML = "";
  for (i = 0; i < countISP + 1; i++) {
    if (!valsISP[i]) valsISP[i] = 0.5;
    ispList.innerHTML +=
      "<li id='ispli" +
      i +
      "'><div class='input-group'><input id='isp" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 max=1 step=0.1 class='form-inline' value='" +
      valsISP[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delIdlerSPMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  feather.replace();
}

function delIdlerSPMode(liId) {
  var countISP = ispList.childElementCount;
  var valsISP = [];
  var j = 0;
  for (i = 0; i < countISP; i++) {
    if (i != liId) {
      valsISP[j] = document.getElementById("isp" + i).value;
      j++;
    }
  }
  ispList.innerHTML = "";
  for (i = 0; i < countISP - 1; i++) {
    document.getElementById("isp-sources").innerHTML +=
      "<li id='ispli" +
      i +
      "'><div class='input-group'><input id='isp" +
      i +
      "' class='form-control form-control-light w-5' type='number' style='width: 30%; display:inline;margin-right: 2%;' min=0 max=1 step=0.1  class='form-inline' value='" +
      valsISP[i] +
      "' aria-label='Thermal'><div class='input-group-append'><span class='input-group-text' title='averge number of photons'> ph </span></div><a style='padding:5px' class='nav-link' href='#' title='Remove this source.' onclick='delIdlerSPMode(&#39;" +
      i +
      "&#39;);return false;'><span style='margin:5%' data-feather='minus-circle'></span></a></div></li>";
  }
  ispn -= 1;
  feather.replace();
  //runCcode();
}

function CopyJPDToClipboard() {
  document.getElementById("datatsv").value = tsvJPDValues;
  document.getElementById("datatsv").select();
  document.execCommand("copy");
}

function CopyRPDToClipboard() {
  document.getElementById("datatsv").value = tsvRPDValues;
  document.getElementById("datatsv").select();
  document.execCommand("copy");
}

function CopyFitToClipboard() {
  document.getElementById("datatsv").value = tsvFitValues;
  document.getElementById("datatsv").select();
  document.execCommand("copy");
}

function hideID(xid) {
  var x = document.getElementById(xid);
  if (x.style.display === "none") {
    x.style.display = "inline";
  } else {
    x.style.display = "none";
  }
}

function hideonlyID(xid) {
  var x = document.getElementById(xid);
  x.style.display = "none";
}

function showonlyID(xid) {
  var x = document.getElementById(xid);
  x.style.display = "inline";
}
