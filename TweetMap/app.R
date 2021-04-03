#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
packages<-c("shiny","dplyr","stringr","tidyr","lubridate","TTR","dygraphs","leaflet", "rgdal","sp","rgeos",
            "shinydashboard", "shinythemes","rjson", "xts", "RColorBrewer","readr")
lapply(packages, library,character.only = TRUE)

tweets_geo <- read_csv("tweets_geo.csv") 
tweets_geo <- tweets_geo%>% drop_na()
tweets_geo$TweetId<-as.character(tweets_geo$TweetId)
#transform the df to json 
coords = data.frame(tweets_geo$lon, tweets_geo$lat)
wgs84 = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
tweets_geo_spdf = SpatialPointsDataFrame(coords, tweets_geo, proj4string = wgs84)
spToGeoJSON <- function(x){
    # Return sp spatial object as geojson by writing a temporary file.
    # It seems the only way to convert sp objects to geojson is 
    # to write a file with OGCGeoJSON driver and read the file back in.
    # The R process must be allowed to write and delete temporoary files.
    #tf<-tempfile('tmp',fileext = '.geojson')
    tf<-tempfile()
    writeOGR(x, tf,layer = "geojson", driver = "GeoJSON")
    js <- paste(readLines(tf), collapse=" ")
    file.remove(tf)
    return(js)
}
tweets_geoJSON<-fromJSON(spToGeoJSON(tweets_geo_spdf))

# Define UI for application that draws a histogram
ui <- dashboardPage(
    dashboardHeader(title = "Tweet Exploring"),
    dashboardSidebar(
        sidebarMenu(
            menuItem("About", tabName = "About", icon = icon("book")),
            menuItem("Golbal Map", tabName = "globalmapdata", icon = icon("map")),
            menuItem("Another Tab", tabName = "statecomp", icon = icon("map"))
        )
    ),
    dashboardBody(
        # tags$script(
        #   '$(".sidebar-toggle").on("click", function() { $(this).trigger("shown"); });'
        # )
        #leaflet
        tags$head(tags$link(rel = "stylesheet", type = "text/css",
                            href = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/leaflet.css")),
        tags$script(src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/leaflet.js"),
        # 
        # # load D3.js library
        # tags$script(src="//d3js.org/d3.v4.min.js"),
        # # load billboard library
        # tags$head(tags$link(rel = "stylesheet", type = "text/css", href = "billboard.min.css")),
        # tags$script(src="billboard.min.js"),
        # 
        # ###classybrew is an automatic classifier see https://github.com/tannerjt/classybrew for source code
        # tags$script(src="classybrew.js"),
        
        tabItems(
            #About the app - create an html document to reference  rather than clutter up the shiny app
            tabItem(tabName = "About",includeHTML("about.html")),
            ##INFO BOXES AND MAP OF US AND DYGRAPH PLOTS  
            tabItem(tabName = "globalmapdata",
                    #### US INFO BOXES
                    #MAP and Dygraphs 
                    box(title= strong(textOutput("MapTitle")), width=8,
                        tags$div(id="mapdiv", style="width: 100%; height: 300px;")),
                    box(width=4,strong("Date: "), textOutput("TwitterDate"), 
                        strong("Tweet Text:"),textOutput("TwitterText"),
                        strong("Found: "),textOutput("FoundWord"),
                        strong("Geo Location: "),textOutput("GeoLocation")),
                    
                        #tags$syle(src='lege.css'),#
                    tags$script(src="latest_map_leaflet.js")
                        # leafletOutput("tweetMap")
                        
                    ),
            
            ######## STATE COMPARISON TAB
            tabItem(tabName = "statecomp")
            ) #tab item end
        ) #dasboard body end
)#ui end

# Define server logic required to draw a histogram
server <- function(input, output, session) {
    output$MapTitle <-  renderText({ paste0("Tweet Map: "
                                            # month(as.Date(max(dfUSOverall$Date))),"/",
                                            # day(as.Date(max(dfUSOverall$Date))),"/",
                                            # year(as.Date(max(dfUSOverall$Date)))
    )})
    session$sendCustomMessage("load_map_geo_data", tweets_geoJSON)
    
    observeEvent(input$click,
                 {clicked_twitt = as.character(isolate(input$click[1]))
                  tweet <- tweets_geo %>% filter(X1 == clicked_twitt)
                     output$TwitterDate <- renderText({paste0(hour(tweet$Datetime),":",
                                                              minute(tweet$Datetime),":",
                                                              second(tweet$Datetime)," ",
                                                              month(tweet$Datetime),"/",
                                                              day(tweet$Datetime),"/",
                                                              year(tweet$Datetime)
                                                               )})
                     ### twitter value boxes
                     output$TwitterText <- renderText({tweet$Text})
                     output$FoundWord <- renderText({tweet$FoundWord})
                     output$GeoLocation <- renderText({paste0(tweet$place_name,", ",
                                                              tweet$admin1,", ", 
                                                              tweet$country_code3)})
})
    # output$tweetMap = renderLeaflet({
    #     
    #     map <- leaflet(tweets_geo, leafletOptions(zoomSnap = 0.25, minZoom = 5))%>% 
    #         addTiles() %>% 
    #         addCircleMarkers()
    # 
    #     
    #     map})
}

# Run the application 
shinyApp(ui = ui, server = server)
