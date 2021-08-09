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
console.log("added osm");
tweetData = [];

var geojsonMarkerOptions = {
    radius: 4,
    fillColor: "#ff7800",
    color: "#000",
    weight: 1,
    opacity: 1,
    fillOpacity: 0.8
};
console.log("maker opts: "+ geojsonMarkerOptions);
var geojson;
var markers;

var info = L.control();

info.onAdd = function (map) {
    this._div = L.DomUtil.create('div', 'info'); // create a div with a class "info"
    this.update();
    console.log("info opts: "+ this);
    return this._div;
    
};

// method that we will use to update the control based on feature properties passed
info.update = function (props) {
    this._div.innerHTML = '<h4>Tweet Info</h4>' +  (props ?
        '<b>Date: </b>' + props.Datetime + '<br/>' +
        '<b>Informativeness O is yes: </b>' + props.target + '<br/>' +
        '<b>Tweet Text: </b>' + props.Text + '<br/>' + 
        '<b>Found: </b>' + props.word + '<br/>' + 
        '<b>Geo Location: </b>' + props.place_name +"," + props.admin1 +"," + props.country_code3
        : 'click for tweet info');

        
};
info.setPosition("topright");
info.addTo(mymap);

function mouse_click (e)
{
  console.log("Clicked: "+ e)
var layer = e.target;
info.update(layer.feature.properties);


  }

function bbox (msg){
  console.log("Bounds: ", msg);
  mymap.flyToBounds(msg);
};

function handle_map_geo_data (msg){
    try { mymap.removeLayer(markers); 
      console.log("removed :" + markers);
    }
  
  catch (nogeojson) { console.log("No data to remove: ",nogeojson);}
    
    console.log("handle_map_geo_data : Start");
    console.log("handle_map_geo_data : message\n" + msg);            
  tweetData = msg;
  console.log("Twiter Data: ", tweetData);
  
  function onEachFeature(feature, layer) {
        layer.options.myTweetID = feature.properties.X1;
        layer.on({
          click: mouse_click
        });

    }; 
//var markers = L.markerClusterGroup();   

function pointToLayer(feature, latlng) {
return L.circleMarker(latlng, geojsonMarkerOptions);
  //return markers.addLayer(pt);
};
    
geojson = L.geoJSON(tweetData, {
    pointToLayer: pointToLayer,
    onEachFeature:onEachFeature} );
markers  = L.markerClusterGroup();
markers.addLayer(geojson).addTo(mymap);
}

Shiny.addCustomMessageHandler("load_map_geo_data", handle_map_geo_data);
console.log("recieved data from shiny" + handle_map_geo_data);
Shiny.addCustomMessageHandler("bounds", bbox);
console.log("recieved data from shiny" + bbox);