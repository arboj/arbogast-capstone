window.onload = function() {
    var anchors = document.getElementsByTagName('li');
    for(var i = 0; i < anchors.length; i++) {
        var anchor = anchors[i];

        anchor.onclick = function() {
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));},25);
        };
    }

};
console.log("Start");


var mymap = L.map('mapdiv',{
  center: [0,0], 
  zoom: 2,
  zoomSnap: 0.1
});

var osmlayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', 
                           {maxZoom: 20,
                           attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'});
osmlayer.addTo(mymap);
tweetData = [];

var geojsonMarkerOptions = {
    radius: 4,
    fillColor: "#ff7800",
    color: "#000",
    weight: 1,
    opacity: 1,
    fillOpacity: 0.8
};

var geojson;

/*
var info = L.control();

info.onAdd = function (map) {
    this._div = L.DomUtil.create('div', 'info'); // create a div with a class "info"
    this.update();
    return this._div;
};

// method that we will use to update the control based on feature properties passed
info.update = function (props) {
    this._div.innerHTML = '<h4>Tweet Info</h4>' +  (props ?
        '<b>' + props.name + '</b><br />' + props.density + ' people / mi<sup>2</sup>'
        : 'Hover over a state');
};

info.addTo(mymap);*/

function mouse_click (e)
{

console.log(e);
var TWID = this.options.myTweetID;
  console.log(TWID);
  //popup("You clicked the marker at Call Sign: " +  ID  + "IN THE STATE OF:" +  STATE + "Station: " + STATION );
  // send this ID to the shiny server code through messaging 
  Shiny.onInputChange("click", [TWID, Math.random()]);
  console.log("click", [TWID, Math.random()]);
  }

function handle_map_geo_data (msg){
  tweetData = msg;
  console.log("Twiter Data: ", tweetData);
  
  function onEachFeature(feature, layer) {
        layer.options.myTweetID = feature.properties.X1;
        layer.on({
          click: mouse_click
        });

    }; 
    
function pointToLayer(feature, latlng) {
  return L.circleMarker(latlng, geojsonMarkerOptions);
};
    
geojson = L.geoJSON(tweetData, {
    pointToLayer: pointToLayer,
    onEachFeature:onEachFeature} ).addTo(mymap);
}

Shiny.addCustomMessageHandler("load_map_geo_data", handle_map_geo_data);