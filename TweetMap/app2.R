# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
library(shiny)
library(shinyTime)
library(dplyr)
library(stringr)
library(lubridate)
library(rgdal)
library(sp)
library(rgeos)
library(shinydashboard)
library(rjson)
library(RColorBrewer)
library(readr)
library(tm)
library(wordcloud)

countries <- read_csv("Countries - Countries.csv")
countrieslist <- countries[c('Country','alpha3')]
firstOrderAdmin <- read_csv('Admin_ones.csv')


wgs84 = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")


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

# Define UI for application that draws a histogram
ui <- dashboardPage(
    dashboardHeader(title = "Natural Disaster Tweet Mapper",
                    titleWidth = 350),
    dashboardSidebar(
      width = 350,
        sidebarMenu(
            menuItem("About", tabName = "About", icon = icon("book")),
            
            menuItem("Map Mentioned Locations", tabName = "map", icon = icon("globe")),
            menuItem("Map Data Filters", icon = icon("filter"),
                     
                     fluidRow(column(width = 6,
                              timeInput("startTime", "Start Time: ", value = NULL, 
                               seconds = FALSE, minute.steps = 15)),
                              column(width = 6,
                              timeInput("endTime", "End Time: ", value = NULL, seconds = FALSE,
                                        minute.steps = 15))),
                     dateRangeInput( "dates", "Date Range: ",
                                     start = "2021-07-31", end = "2021-08-01",
                                     min = min(tweets_geo$Datetime), max = Sys.Date(),
                                     format = "yyyy-mm-dd", startview = "month",
                                     weekstart = 0,language = "en",
                                     separator = " to ",width = NULL, autoclose = TRUE),
                     checkboxGroupInput("DisasterType", "Disaster Type to show:",
                                        c("Fire and Heat Related" = "fire",
                                          "Geological/Movement" ="geomovement" ,
                                          "Severe Weather" = "severeweather" ,
                                          "Tropical Weather" = "tropical",
                                          "Floods" = "flood")),
                     fluidRow(column(width = 6, actionButton("HORY", "Filter Search")), 
                              column(width = 6, actionButton("ALL", "No Event Filter"))),
                     fluidRow(column(width = 6, actionButton("worldData", "Search World Wide")), 
                              column(width = 6, actionButton("Reset", "Reset to Country"))),
              
                           selectInput("CountrySelect", "Select Country: ",
                                       choices = countrieslist$Country),
                           actionButton("CountrySearch", "1. Search By Country:"),
                           selectInput("FirstOrderAdminSelect",
                                       "Select First Order Admin: ",
                                       choices=character(0),
                                       selected=character(0)),
                           actionButton("Drill", "2: Drill to 1st Level Admin")
                           )
            ,
            menuItem("Data Table", tabName = "dataTable", icon = icon("table"))
        )
    ),
    dashboardBody(
        tabItems(
            #About the app - create an html document to reference  rather than clutter up the shiny app
            tabItem(tabName = "About",includeHTML("about.html")),
            ##INFO BOXES AND MAP 
            tabItem( tags$head(tags$style(HTML(
                    ".info {
                        padding: 6px 8px;
                        font: 14px/16px Arial, Helvetica, sans-serif;
                        background: white;
                        background: rgba(255,255,255,0.8);
                        box-shadow: 0 0 15px rgba(0,0,0,0.2);
                        border-radius: 5px;
                        width: 250px
                    }
                    .info h4 {
                        margin: 0 0 5px;
                        color: #777;
                    }"
                )),
                tags$link(rel = "stylesheet", type = "text/css",
                          href = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/leaflet.css"),
                tags$link(rel = "stylesheet", type = "text/css",
                          href="https://unpkg.com/leaflet.markercluster@1.3.0/dist/MarkerCluster.css"),
                tags$link(rel = "stylesheet", type = "text/css",          
                          href="https://unpkg.com/leaflet.markercluster@1.3.0/dist/MarkerCluster.Default.css")),
                tags$script(src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.1/leaflet.js"),
                tags$script(src="https://unpkg.com/leaflet.markercluster@1.3.0/dist/leaflet.markercluster.js"),
              tabName = "map",
                column(width = 8,
                       box(width= 12,
                           title= strong(textOutput("MapTitle")),
                           tags$div(id="mapdiv", style="width: 100%; height: 700px;"),
                           tags$script(src="latest_map_leaflet.js"))
                       
                ),
                column(width= 4,box(width= 12,

                    sliderInput("maxnumw","Maximum number of words:",min=1, max=100, step=5, value=25),
                    sliderInput("minfreqw","Minimum frequency:",min=1, max=500, step=10, value=200),),
                box(width= 12,plotOutput("wordc"))
               ),
               ),
            tabItem(tabName = "dataTable",downloadButton("download1"),dataTableOutput("mappedDataTable"))
            ) #tab items end
        ) #dasboard body end
)#ui end

server <- function(input, output, session) {
  
  observeEvent(input$ALL,{
    observeEvent(input$worldData,{
      
      values <- reactiveValues(start = as.character(input$dates[1]),end= as.character(input$dates[2]))
      hrs <- reactiveValues(start = as.character(strftime(input$startTime,"%T")),
                             end = as.character(strftime(input$endTime,"%T")))
      
      tweets_geo_sub <- read_csv('alltweet.csv',
                             col_select = -c(target,spans,country_predicted,
                                                            country_conf,feature_class),
                             col_types = cols(TweetId = col_character())) %>%
                             filter(Datetime >= strptime(paste(values$start,hrs$start), 
                                                         format="%Y-%m-%d %H:%M:%S",tz="GMT")) %>% 
                             filter(Datetime < strptime(paste(values$end,hrs$end),
                                                        format="%Y-%m-%d %H:%M:%S",tz="GMT"))

      
      output$MapTitle <-  renderText({ paste0("Tweet Map - World Wide: ",
                                               month(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                               day(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                               year(as.Date(min(tweets_geo_sub$Datetime))) ," ",
                                               str_sub(as.character(min(tweets_geo_sub$Datetime)),
                                                       start= -8),"UTC", ' to ',
                                               month(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                               day(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                               year(as.Date(max(tweets_geo_sub$Datetime)))," ",
                                               str_sub(as.character(max(tweets_geo_sub$Datetime)),
                                                       start= -8),"UTC"
      )})
      output$mappedDataTable<-renderDataTable(tweets_geo_sub)
      output$download1 <- downloadHandler(
        filename = function() {
          paste0("Export.csv")
        },
        content = function(file) {
          write.csv(tweets_geo_sub, file)
        }
      )
      coords = data.frame(tweets_geo_sub$lon, tweets_geo_sub$lat)
      tweets_geo_spdf = SpatialPointsDataFrame(coords, tweets_geo_sub, proj4string = wgs84)       
      tweets_geoJSON<-fromJSON(spToGeoJSON(tweets_geo_spdf))
      
      session$sendCustomMessage("load_map_geo_data", tweets_geoJSON)
      
      d1<-(bbox(tweets_geo_spdf)[4]-bbox(tweets_geo_spdf)[2])*.2
      d2<-(bbox(tweets_geo_spdf)[3]-bbox(tweets_geo_spdf)[1])*.2
      vector1 <- c(bbox(tweets_geo_spdf)[2]-d1,bbox(tweets_geo_spdf)[4]+d1)
      vector2 <- c(bbox(tweets_geo_spdf)[1]-d2,bbox(tweets_geo_spdf)[3]+d2)

      box<-array(c(vector1,vector2),dim = c(2,2))

      session$sendCustomMessage("bounds", box)
      
      output$wordc<- renderPlot ({
        linesg<-tweets_geo_sub$ptext
        docs <- Corpus(VectorSource(linesg))
        
        dtm2 <- TermDocumentMatrix(docs)
        m2 <- as.matrix(dtm2)
        v2 <- sort(rowSums(m2),decreasing=TRUE)
        d2 <- data.frame(word = names(v2),freq=v2)
        wordcloud(words = d2$word, freq = d2$freq, min.freq = input$minfreqw,
                  max.words=input$maxnumw, random.order=FALSE, rot.per=0,
                  colors=brewer.pal(8, "Dark2"))
      })
      
    })
    observeEvent(input$CountrySearch, {
      selectedCountry<-isolate(input$CountrySelect)
      selectedNation<<-toString(countrieslist %>% filter(Country==selectedCountry)%>%select(alpha3))
      
      values <- reactiveValues(start = as.character(input$dates[1]),end= as.character(input$dates[2]))
      hrs <- reactiveValues(start = as.character(strftime(input$startTime,"%T")), 
                            end = as.character(strftime(input$endTime,"%T")))
      
      tweets_geo_sub <- read_csv('alltweet.csv',
                                 col_select = -c(target,spans,country_predicted,
                                                 country_conf,feature_class),
                                 col_types = cols(TweetId = col_character())) %>%
        filter(Datetime >= strptime(paste(values$start,hrs$start), 
                                    format="%Y-%m-%d %H:%M:%S",tz="GMT")) %>% 
        filter(Datetime < strptime(paste(values$end,hrs$end),
                                   format="%Y-%m-%d %H:%M:%S",tz="GMT")) %>%
        filter(country_code3 ==selectedNation)
      
      
      
      output$MapTitle <-  renderText({ paste0("Tweet Map - ",selectedCountry,": ",
                                              month(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                              day(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                              year(as.Date(min(tweets_geo_sub$Datetime))) ," ",
                                              str_sub(as.character(min(tweets_geo_sub$Datetime)), 
                                                      start= -8),"UTC", ' to ',
                                              month(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                              day(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                              year(as.Date(max(tweets_geo_sub$Datetime)))," ",
                                              str_sub(as.character(max(tweets_geo_sub$Datetime)), 
                                                      start= -8),"UTC"
                                              
      )})
      firstOrderAdminlist<<- firstOrderAdmin %>% 
        filter(country_code3 == selectedNation) %>% select(admin1)
      updateSelectInput(session,"FirstOrderAdminSelect",
                        "Select First Order Admin: ",
                        choices = firstOrderAdminlist)
      
      
      
      observeEvent(input$Drill,{
        values <- reactiveValues(start = as.character(input$dates[1]),end= as.character(input$dates[2]))
        hrs <- reactiveValues(start = as.character(strftime(input$startTime,"%T")), 
                              end = as.character(strftime(input$endTime,"%T")))
        
        tweets_geo_sub <-subset(tweets_geo, tweets_geo$Datetime >=
                                     strptime(paste(values$start,hrs$start),format="%Y-%m-%d %H:%M:%S",tz="GMT") &
                                     tweets_geo$Datetime < 
                                     strptime(paste(values$end,hrs$end),format="%Y-%m-%d %H:%M:%S",tz="GMT")&
                                     tweets_geo$country_code3 ==selectedNation & 
                                     tweets_geo$admin1 ==input$FirstOrderAdminSelect)
        
        
        
        output$MapTitle <-  renderText({ paste0("Tweet Map - ",input$FirstOrderAdminSelect, " ",selectedCountry,": ",
                                                month(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                                day(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                                year(as.Date(min(tweets_geo_sub$Datetime))) ," ",
                                                str_sub(as.character(min(tweets_geo_sub$Datetime)), 
                                                        start= -8),"UTC", ' to ',
                                                month(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                                day(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                                year(as.Date(max(tweets_geo_sub$Datetime)))," ",
                                                str_sub(as.character(max(tweets_geo_sub$Datetime)), 
                                                        start= -8),"UTC"
        )})
        output$mappedDataTable<-renderDataTable(tweets_geo_sub)
       
         output$download1 <- downloadHandler(
          filename = function() {
            paste0("Export.csv")
          },
          content = function(file) {
            write.csv(tweets_geo_sub, file)
          }
        )
         
        coords = data.frame(tweets_geo_sub$lon, tweets_geo_sub$lat)
        tweets_geo_spdf = SpatialPointsDataFrame(coords, tweets_geo_sub, proj4string = wgs84)
        tweets_geoJSON<-fromJSON(spToGeoJSON(tweets_geo_spdf))
        
        session$sendCustomMessage("load_map_geo_data", tweets_geoJSON)
        d1<-(bbox(tweets_geo_spdf)[4]-bbox(tweets_geo_spdf)[2])*.2
        d2<-(bbox(tweets_geo_spdf)[3]-bbox(tweets_geo_spdf)[1])*.2
        vector1 <- c(bbox(tweets_geo_spdf)[2]-d1,bbox(tweets_geo_spdf)[4]+d1)
        vector2 <- c(bbox(tweets_geo_spdf)[1]-d2,bbox(tweets_geo_spdf)[3]+d2)
        box<-array(c(vector1,vector2),dim = c(2,2))
        
        
        
        session$sendCustomMessage("bounds", box)
        
        output$wordc<- renderPlot ({
          lines<-tweets_geo_sub$ptext
          docs <- Corpus(VectorSource(lines))
          
          dtm2 <- TermDocumentMatrix(docs)
          m2 <- as.matrix(dtm2)
          v2 <- sort(rowSums(m2),decreasing=TRUE)
          d2 <- data.frame(word = names(v2),freq=v2)
          wordcloud(words = d2$word, freq = d2$freq, min.freq = input$minfreqw, 
                    max.words=input$maxnumw, random.order=FALSE, rot.per=0, 
                    colors=brewer.pal(8, "Dark2"))
        })
        
        observeEvent(input$Reset,{
          values <- reactiveValues(start = as.character(input$dates[1]),end= as.character(input$dates[2]))
          hrs <- reactiveValues(start = as.character(strftime(input$startTime,"%T")), 
                                end = as.character(strftime(input$endTime,"%T")))
          

          tweets_geo_sub <- read_csv('alltweet.csv',
                                     col_select = -c(target,spans,country_predicted,
                                                     country_conf,feature_class),
                                     col_types = cols(TweetId = col_character())) %>%
            filter(Datetime >= strptime(paste(values$start,hrs$start), 
                                        format="%Y-%m-%d %H:%M:%S",tz="GMT")) %>% 
            filter(Datetime < strptime(paste(values$end,hrs$end),
                                       format="%Y-%m-%d %H:%M:%S",tz="GMT")) %>%
            filter(country_code3 ==selectedNation)
          
          output$MapTitle <-  renderText({ paste0("Tweet Map - ",selectedCountry,": ",
                                                  month(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                                  day(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                                  year(as.Date(min(tweets_geo_sub$Datetime))) ," ",
                                                  str_sub(as.character(min(tweets_geo_sub$Datetime)), 
                                                          start= -8),"UTC", ' to ',
                                                  # hour(min(tweets_geo_sub$Datetime)),":",
                                                  # minute(min(tweets_geo_sub$Datetime)),":",
                                                  # second(min(tweets_geo_sub$Datetime)),"UTC",' to ',
                                                  month(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                                  day(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                                  year(as.Date(max(tweets_geo_sub$Datetime)))," ",
                                                  str_sub(
                                                    as.character(max(tweets_geo_sub$Datetime)), 
                                                    start= -8),"UTC"
                                                  # hour(max(tweets_geo_sub$Datetime)),":",
                                                  # minute(max(tweets_geo_sub$Datetime)),":",
                                                  # second(max(tweets_geo_sub$Datetime)),"UTC"
                                                  
                                                  
          )})
          
          firstOrderAdminlist<<- firstOrderAdmin %>% 
            filter(country_code3 == selectedNation) %>% select(admin1)
          updateSelectInput(session,"FirstOrderAdminSelect",
                            "Select First Order Admin: ",
                            choices = firstOrderAdminlist)
          coords = data.frame(tweets_geo_sub$lon, tweets_geo_sub$lat)
          tweets_geo_spdf = SpatialPointsDataFrame(coords, tweets_geo_sub, proj4string = wgs84)
          tweets_geoJSON<-fromJSON(spToGeoJSON(tweets_geo_spdf))
          
          output$mappedDataTable<-renderDataTable(tweets_geo_sub)
          
          output$download1 <- downloadHandler(
            filename = function() {
              paste0("Export.csv")
            },
            content = function(file) {
              write.csv(tweets_geo_sub, file)
            }
          )
          
          session$sendCustomMessage("load_map_geo_data", tweets_geoJSON)
          
          d1<-(bbox(tweets_geo_spdf)[4]-bbox(tweets_geo_spdf)[2])*.2
          d2<-(bbox(tweets_geo_spdf)[3]-bbox(tweets_geo_spdf)[1])*.2
          vector1 <- c(bbox(tweets_geo_spdf)[2]-d1,bbox(tweets_geo_spdf)[4]+d1)
          vector2 <- c(bbox(tweets_geo_spdf)[1]-d2,bbox(tweets_geo_spdf)[3]+d2)
          
          box<-array(c(vector1,vector2),dim = c(2,2))
          
          session$sendCustomMessage("bounds", box)
          output$wordc<- renderPlot ({
            lines<-tweets_geo_sub$ptext
            docs <- Corpus(VectorSource(lines))
            
            dtm2 <- TermDocumentMatrix(docs)
            m2 <- as.matrix(dtm2)
            v2 <- sort(rowSums(m2),decreasing=TRUE)
            d2 <- data.frame(word = names(v2),freq=v2)
            wordcloud(words = d2$word, freq = d2$freq, min.freq = input$minfreqw, 
                      max.words=input$maxnumw, random.order=FALSE, rot.per=0, 
                      colors=brewer.pal(8, "Dark2"))
          })
          
        })
        
        
      })
      
      coords = data.frame(tweets_geo_sub$lon, tweets_geo_sub$lat)
      tweets_geo_spdf = SpatialPointsDataFrame(coords, tweets_geo_sub, proj4string = wgs84)
      tweets_geoJSON<-fromJSON(spToGeoJSON(tweets_geo_spdf))
      
      output$mappedDataTable<-renderDataTable(tweets_geo_sub)
      
      output$download1 <- downloadHandler(
        filename = function() {
          paste0("Export.csv")
        },
        content = function(file) {
          write.csv(tweets_geo_sub, file)
        }
      )
      
      session$sendCustomMessage("load_map_geo_data", tweets_geoJSON)
      
      d1<-(bbox(tweets_geo_spdf)[4]-bbox(tweets_geo_spdf)[2])*.2
      d2<-(bbox(tweets_geo_spdf)[3]-bbox(tweets_geo_spdf)[1])*.2
      vector1 <- c(bbox(tweets_geo_spdf)[2]-d1,bbox(tweets_geo_spdf)[4]+d1)
      vector2 <- c(bbox(tweets_geo_spdf)[1]-d2,bbox(tweets_geo_spdf)[3]+d2)
      
      box<-array(c(vector1,vector2),dim = c(2,2))
      
      session$sendCustomMessage("bounds", box)
      output$wordc<- renderPlot ({
        lines<-tweets_geo_sub$ptext
        docs <- Corpus(VectorSource(lines))
        
        dtm2 <- TermDocumentMatrix(docs)
        m2 <- as.matrix(dtm2)
        v2 <- sort(rowSums(m2),decreasing=TRUE)
        d2 <- data.frame(word = names(v2),freq=v2)
        wordcloud(words = d2$word, freq = d2$freq, min.freq = input$minfreqw, 
                  max.words=input$maxnumw, random.order=FALSE, rot.per=0, 
                  colors=brewer.pal(8, "Dark2"))
      })
      
    })
    })
  
  
  observeEvent(input$HORY, { listl<-length(input$DisasterType)
  
  observeEvent(input$worldData,{
    
    
    values <- reactiveValues(start = as.character(input$dates[1]),end= as.character(input$dates[2]))
    hrs <- reactiveValues(start = as.character(strftime(input$startTime,"%T")),
                          end = as.character(strftime(input$endTime,"%T")))
    
    
    tweets_geo_sub <- read_csv('alltweet.csv',
                               col_select = -c(target,spans,country_predicted,
                                               country_conf,feature_class),
                               col_types = cols(TweetId = col_character())) %>%
      filter(Datetime >= strptime(paste(values$start,hrs$start), 
                                  format="%Y-%m-%d %H:%M:%S",tz="GMT")) %>% 
      filter(Datetime < strptime(paste(values$end,hrs$end),
                                 format="%Y-%m-%d %H:%M:%S",tz="GMT")) 
    if (listl == 1){tweets_geo_sub <-subset(tweets_geo_sub,
                                            ((tweets_geo_sub[input$DisasterType[1]] == TRUE))) }
    else if(listl == 2){tweets_geo_sub<-subset(tweets_geo_sub, ((tweets_geo_sub[input$DisasterType[1]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[2]] == TRUE)))}
    else if(listl == 3){tweets_geo_sub<-subset(tweets_geo_sub,((tweets_geo_sub[input$DisasterType[1]]  == TRUE |
                                                                  (tweets_geo_sub[input$DisasterType[2]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[3]] == TRUE))))}
    else if(listl == 4){tweets_geo_sub<-subset(tweets_geo_sub,((tweets_geo_sub[input$DisasterType[1]]  == TRUE |
                                                                  (tweets_geo_sub[input$DisasterType[2]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[3]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[4]] == TRUE))))}
    else if(listl == 5){tweets_geo_sub<-subset(tweets_geo_sub,((tweets_geo_sub[input$DisasterType[1]]  == TRUE |
                                                                  (tweets_geo_sub[input$DisasterType[2]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[3]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[4]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[5]] == TRUE))))}
    output$MapTitle <-  renderText({ paste0("Tweet Map - World Wide: ",
                                            month(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                            day(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                            year(as.Date(min(tweets_geo_sub$Datetime))) ," ",
                                            str_sub(as.character(min(tweets_geo_sub$Datetime)),
                                                    start= -8),"UTC", ' to ',
                                            month(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                            day(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                            year(as.Date(max(tweets_geo_sub$Datetime)))," ",
                                            str_sub(as.character(max(tweets_geo_sub$Datetime)),
                                                    start= -8),"UTC"
    )})
    output$mappedDataTable<-renderDataTable(tweets_geo_sub)
    
    output$download1 <- downloadHandler(
      filename = function() {
        paste0("Export.csv")
      },
      content = function(file) {
        write.csv(tweets_geo_sub, file)
      }
    )
    
    coords = data.frame(tweets_geo_sub$lon, tweets_geo_sub$lat)
    tweets_geo_spdf = SpatialPointsDataFrame(coords, tweets_geo_sub, proj4string = wgs84)
    tweets_geoJSON<-fromJSON(spToGeoJSON(tweets_geo_spdf))
    
    
    session$sendCustomMessage("load_map_geo_data", tweets_geoJSON)
    
    d1<-(bbox(tweets_geo_spdf)[4]-bbox(tweets_geo_spdf)[2])*.2
    d2<-(bbox(tweets_geo_spdf)[3]-bbox(tweets_geo_spdf)[1])*.2
    vector1 <- c(bbox(tweets_geo_spdf)[2]-d1,bbox(tweets_geo_spdf)[4]+d1)
    vector2 <- c(bbox(tweets_geo_spdf)[1]-d2,bbox(tweets_geo_spdf)[3]+d2)
    
    box<-array(c(vector1,vector2),dim = c(2,2))
    
    session$sendCustomMessage("bounds", box)
    
    output$wordc<- renderPlot ({
      linesg<-tweets_geo_sub$ptext
      docs <- Corpus(VectorSource(linesg))
      
      dtm2 <- TermDocumentMatrix(docs)
      m2 <- as.matrix(dtm2)
      v2 <- sort(rowSums(m2),decreasing=TRUE)
      d2 <- data.frame(word = names(v2),freq=v2)
      wordcloud(words = d2$word, freq = d2$freq, min.freq = input$minfreqw,
                max.words=input$maxnumw, random.order=FALSE, rot.per=0,
                colors=brewer.pal(8, "Dark2"))
    })
    
  })
  
  observeEvent(input$CountrySearch, {
    selectedCountry<-isolate(input$CountrySelect)
    selectedNation<<-toString(countrieslist %>% filter(Country==selectedCountry)%>%select(alpha3))
    
    values <- reactiveValues(start = as.character(input$dates[1]),end= as.character(input$dates[2]))
    hrs <- reactiveValues(start = as.character(strftime(input$startTime,"%T")), 
                          end = as.character(strftime(input$endTime,"%T")))
    
    
    tweets_geo_sub <- read_csv('alltweet.csv',
                               col_select = -c(target,spans,country_predicted,
                                               country_conf,feature_class),
                               col_types = cols(TweetId = col_character())) %>%
      filter(Datetime >= strptime(paste(values$start,hrs$start), 
                                  format="%Y-%m-%d %H:%M:%S",tz="GMT")) %>% 
      filter(Datetime < strptime(paste(values$end,hrs$end),
                                 format="%Y-%m-%d %H:%M:%S",tz="GMT")) %>%
      filter(country_code3 ==selectedNation)
    if (listl == 1){tweets_geo_sub <-subset(tweets_geo_sub,
                                            ((tweets_geo_sub[input$DisasterType[1]] == TRUE))) }
    else if(listl == 2){tweets_geo_sub<-subset(tweets_geo_sub, ((tweets_geo_sub[input$DisasterType[1]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[2]] == TRUE)))}
    else if(listl == 3){tweets_geo_sub<-subset(tweets_geo_sub,((tweets_geo_sub[input$DisasterType[1]]  == TRUE |
                                                                  (tweets_geo_sub[input$DisasterType[2]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[3]] == TRUE))))}
    else if(listl == 4){tweets_geo_sub<-subset(tweets_geo_sub,((tweets_geo_sub[input$DisasterType[1]]  == TRUE |
                                                                  (tweets_geo_sub[input$DisasterType[2]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[3]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[4]] == TRUE))))}
    else if(listl == 5){tweets_geo_sub<-subset(tweets_geo_sub,((tweets_geo_sub[input$DisasterType[1]]  == TRUE |
                                                                  (tweets_geo_sub[input$DisasterType[2]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[3]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[4]] == TRUE) |
                                                                  (tweets_geo_sub[input$DisasterType[5]] == TRUE))))}
    
    output$MapTitle <-  renderText({ paste0("Tweet Map - ",selectedCountry,": ",
                                            month(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                            day(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                            year(as.Date(min(tweets_geo_sub$Datetime))) ," ",
                                            str_sub(as.character(min(tweets_geo_sub$Datetime)), 
                                                    start= -8),"UTC", ' to ',
                                            month(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                            day(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                            year(as.Date(max(tweets_geo_sub$Datetime)))," ",
                                            str_sub(as.character(max(tweets_geo_sub$Datetime)), 
                                                    start= -8),"UTC"
                                            
    )})
    firstOrderAdminlist<<- firstOrderAdmin %>% 
      filter(country_code3 == selectedNation) %>% select(admin1)
    updateSelectInput(session,"FirstOrderAdminSelect",
                      "Select First Order Admin: ",
                      choices = firstOrderAdminlist)
    
    
    
    observeEvent(input$Drill,{
      values <- reactiveValues(start = as.character(input$dates[1]),end= as.character(input$dates[2]))
      hrs <- reactiveValues(start = as.character(strftime(input$startTime,"%T")), 
                            end = as.character(strftime(input$endTime,"%T")))
      
      tweets_geo_sub <- read_csv('alltweet.csv',
                                 col_select = -c(target,spans,country_predicted,
                                                 country_conf,feature_class),
                                 col_types = cols(TweetId = col_character())) %>%
        filter(Datetime >= strptime(paste(values$start,hrs$start), 
                                    format="%Y-%m-%d %H:%M:%S",tz="GMT")) %>% 
        filter(Datetime < strptime(paste(values$end,hrs$end),
                                   format="%Y-%m-%d %H:%M:%S",tz="GMT")) %>%
        filter(country_code3 ==selectedNation) %>%
        filter(admin1 ==input$FirstOrderAdminSelect)
      
      if (listl == 1){tweets_geo_sub <-subset(tweets_geo_sub,
                                              ((tweets_geo_sub[input$DisasterType[1]] == TRUE))) }
      else if(listl == 2){tweets_geo_sub<-subset(tweets_geo_sub, ((tweets_geo_sub[input$DisasterType[1]] == TRUE) |
                                                                    (tweets_geo_sub[input$DisasterType[2]] == TRUE)))}
      else if(listl == 3){tweets_geo_sub<-subset(tweets_geo_sub,((tweets_geo_sub[input$DisasterType[1]]  == TRUE |
                                                                    (tweets_geo_sub[input$DisasterType[2]] == TRUE) |
                                                                    (tweets_geo_sub[input$DisasterType[3]] == TRUE))))}
      else if(listl == 4){tweets_geo_sub<-subset(tweets_geo_sub,((tweets_geo_sub[input$DisasterType[1]]  == TRUE |
                                                                    (tweets_geo_sub[input$DisasterType[2]] == TRUE) |
                                                                    (tweets_geo_sub[input$DisasterType[3]] == TRUE) |
                                                                    (tweets_geo_sub[input$DisasterType[4]] == TRUE))))}
      else if(listl == 5){tweets_geo_sub<-subset(tweets_geo_sub,((tweets_geo_sub[input$DisasterType[1]]  == TRUE |
                                                                    (tweets_geo_sub[input$DisasterType[2]] == TRUE) |
                                                                    (tweets_geo_sub[input$DisasterType[3]] == TRUE) |
                                                                    (tweets_geo_sub[input$DisasterType[4]] == TRUE) |
                                                                    (tweets_geo_sub[input$DisasterType[5]] == TRUE))))}
      
      
      
      output$MapTitle <-  renderText({ paste0("Tweet Map - ",input$FirstOrderAdminSelect, " ",selectedCountry,": ",
                                              month(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                              day(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                              year(as.Date(min(tweets_geo_sub$Datetime))) ," ",
                                              str_sub(as.character(min(tweets_geo_sub$Datetime)), 
                                                      start= -8),"UTC", ' to ',
                                              month(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                              day(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                              year(as.Date(max(tweets_geo_sub$Datetime)))," ",
                                              str_sub(as.character(max(tweets_geo_sub$Datetime)), 
                                                      start= -8),"UTC"
      )})

      output$mappedDataTable<-renderDataTable(tweets_geo_sub)
      
      output$download1 <- downloadHandler(
        filename = function() {
          paste0("Export.csv")
        },
        content = function(file) {
          write.csv(tweets_geo_sub, file)
        }
      )
      
      coords = data.frame(tweets_geo_sub$lon, tweets_geo_sub$lat)
      tweets_geo_spdf = SpatialPointsDataFrame(coords, tweets_geo_sub, proj4string = wgs84)
      tweets_geoJSON<-fromJSON(spToGeoJSON(tweets_geo_spdf))
      
      session$sendCustomMessage("load_map_geo_data", tweets_geoJSON)
      d1<-(bbox(tweets_geo_spdf)[4]-bbox(tweets_geo_spdf)[2])*.2
      d2<-(bbox(tweets_geo_spdf)[3]-bbox(tweets_geo_spdf)[1])*.2
      vector1 <- c(bbox(tweets_geo_spdf)[2]-d1,bbox(tweets_geo_spdf)[4]+d1)
      vector2 <- c(bbox(tweets_geo_spdf)[1]-d2,bbox(tweets_geo_spdf)[3]+d2)
      box<-array(c(vector1,vector2),dim = c(2,2))
      
      
      
      session$sendCustomMessage("bounds", box)
      
      output$wordc<- renderPlot ({
        lines<-tweets_geo_sub$ptext
        docs <- Corpus(VectorSource(lines))
        
        dtm2 <- TermDocumentMatrix(docs)
        m2 <- as.matrix(dtm2)
        v2 <- sort(rowSums(m2),decreasing=TRUE)
        d2 <- data.frame(word = names(v2),freq=v2)
        wordcloud(words = d2$word, freq = d2$freq, min.freq = input$minfreqw, 
                  max.words=input$maxnumw, random.order=FALSE, rot.per=0, 
                  colors=brewer.pal(8, "Dark2"))
      })
      
      observeEvent(input$Reset,{
        values <- reactiveValues(start = as.character(input$dates[1]),end= as.character(input$dates[2]))
        hrs <- reactiveValues(start = as.character(strftime(input$startTime,"%T")), 
                              end = as.character(strftime(input$endTime,"%T")))
        tweets_geo_sub <- read_csv('alltweet.csv',
                                   col_select = -c(target,spans,country_predicted,
                                                   country_conf,feature_class),
                                   col_types = cols(TweetId = col_character())) %>%
          filter(Datetime >= strptime(paste(values$start,hrs$start), 
                                      format="%Y-%m-%d %H:%M:%S",tz="GMT")) %>% 
          filter(Datetime < strptime(paste(values$end,hrs$end),
                                     format="%Y-%m-%d %H:%M:%S",tz="GMT")) %>%
          filter(country_code3 ==selectedNation)
        
        if (listl == 1){tweets_geo_sub <-subset(tweets_geo_sub,
                                                ((tweets_geo_sub[input$DisasterType[1]] == TRUE))) }
        else if(listl == 2){tweets_geo_sub<-subset(tweets_geo_sub, ((tweets_geo_sub[input$DisasterType[1]] == TRUE) |
                                                                      (tweets_geo_sub[input$DisasterType[2]] == TRUE)))}
        else if(listl == 3){tweets_geo_sub<-subset(tweets_geo_sub,((tweets_geo_sub[input$DisasterType[1]]  == TRUE |
                                                                      (tweets_geo_sub[input$DisasterType[2]] == TRUE) |
                                                                      (tweets_geo_sub[input$DisasterType[3]] == TRUE))))}
        else if(listl == 4){tweets_geo_sub<-subset(tweets_geo_sub,((tweets_geo_sub[input$DisasterType[1]]  == TRUE |
                                                                      (tweets_geo_sub[input$DisasterType[2]] == TRUE) |
                                                                      (tweets_geo_sub[input$DisasterType[3]] == TRUE) |
                                                                      (tweets_geo_sub[input$DisasterType[4]] == TRUE))))}
        else if(listl == 5){tweets_geo_sub<-subset(tweets_geo_sub,((tweets_geo_sub[input$DisasterType[1]]  == TRUE |
                                                                      (tweets_geo_sub[input$DisasterType[2]] == TRUE) |
                                                                      (tweets_geo_sub[input$DisasterType[3]] == TRUE) |
                                                                      (tweets_geo_sub[input$DisasterType[4]] == TRUE) |
                                                                      (tweets_geo_sub[input$DisasterType[5]] == TRUE))))}
        
        output$MapTitle <-  renderText({ paste0("Tweet Map - ",selectedCountry,": ",
                                                month(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                                day(as.Date(min(tweets_geo_sub$Datetime))),"/",
                                                year(as.Date(min(tweets_geo_sub$Datetime))) ," ",
                                                str_sub(as.character(min(tweets_geo_sub$Datetime)), 
                                                        start= -8),"UTC", ' to ',
                                                # hour(min(tweets_geo_sub$Datetime)),":",
                                                # minute(min(tweets_geo_sub$Datetime)),":",
                                                # second(min(tweets_geo_sub$Datetime)),"UTC",' to ',
                                                month(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                                day(as.Date(max(tweets_geo_sub$Datetime))),"/",
                                                year(as.Date(max(tweets_geo_sub$Datetime)))," ",
                                                str_sub(
                                                  as.character(max(tweets_geo_sub$Datetime)), 
                                                  start= -8),"UTC"
                                                # hour(max(tweets_geo_sub$Datetime)),":",
                                                # minute(max(tweets_geo_sub$Datetime)),":",
                                                # second(max(tweets_geo_sub$Datetime)),"UTC"
                                                
                                                
        )})
        
        firstOrderAdminlist<<- firstOrderAdmin %>% 
          filter(country_code3 == selectedNation) %>% select(admin1)
        updateSelectInput(session,"FirstOrderAdminSelect",
                          "Select First Order Admin: ",
                          choices = firstOrderAdminlist)
        coords = data.frame(tweets_geo_sub$lon, tweets_geo_sub$lat)
        tweets_geo_spdf = SpatialPointsDataFrame(coords, tweets_geo_sub, proj4string = wgs84)
        tweets_geoJSON<-fromJSON(spToGeoJSON(tweets_geo_spdf))
        
        output$mappedDataTable<-renderDataTable(tweets_geo_sub)
        
        output$download1 <- downloadHandler(
          filename = function() {
            paste0("Export.csv")
          },
          content = function(file) {
            write.csv(tweets_geo_sub, file)
          }
        )
        
        session$sendCustomMessage("load_map_geo_data", tweets_geoJSON)
        
        d1<-(bbox(tweets_geo_spdf)[4]-bbox(tweets_geo_spdf)[2])*.2
        d2<-(bbox(tweets_geo_spdf)[3]-bbox(tweets_geo_spdf)[1])*.2
        vector1 <- c(bbox(tweets_geo_spdf)[2]-d1,bbox(tweets_geo_spdf)[4]+d1)
        vector2 <- c(bbox(tweets_geo_spdf)[1]-d2,bbox(tweets_geo_spdf)[3]+d2)
        
        box<-array(c(vector1,vector2),dim = c(2,2))
        
        session$sendCustomMessage("bounds", box)
        output$wordc<- renderPlot ({
          lines<-tweets_geo_sub$ptext
          docs <- Corpus(VectorSource(lines))
          
          dtm2 <- TermDocumentMatrix(docs)
          m2 <- as.matrix(dtm2)
          v2 <- sort(rowSums(m2),decreasing=TRUE)
          d2 <- data.frame(word = names(v2),freq=v2)
          wordcloud(words = d2$word, freq = d2$freq, min.freq = input$minfreqw, 
                    max.words=input$maxnumw, random.order=FALSE, rot.per=0, 
                    colors=brewer.pal(8, "Dark2"))
        })
        
      })
      
      
    })

    coords = data.frame(tweets_geo_sub$lon, tweets_geo_sub$lat)
    tweets_geo_spdf = SpatialPointsDataFrame(coords, tweets_geo_sub, proj4string = wgs84)
    tweets_geoJSON<-fromJSON(spToGeoJSON(tweets_geo_spdf))
    output$mappedDataTable<-renderDataTable(tweets_geo_sub)
    
    output$download1 <- downloadHandler(
      filename = function() {
        paste0("Export.csv")
      },
      content = function(file) {
        write.csv(tweets_geo_sub, file)
      }
    )
    session$sendCustomMessage("load_map_geo_data", tweets_geoJSON)
    
    d1<-(bbox(tweets_geo_spdf)[4]-bbox(tweets_geo_spdf)[2])*.2
    d2<-(bbox(tweets_geo_spdf)[3]-bbox(tweets_geo_spdf)[1])*.2
    vector1 <- c(bbox(tweets_geo_spdf)[2]-d1,bbox(tweets_geo_spdf)[4]+d1)
    vector2 <- c(bbox(tweets_geo_spdf)[1]-d2,bbox(tweets_geo_spdf)[3]+d2)
    
    box<-array(c(vector1,vector2),dim = c(2,2))
    
    session$sendCustomMessage("bounds", box)
    output$wordc<- renderPlot ({
      lines<-tweets_geo_sub$ptext
      docs <- Corpus(VectorSource(lines))
      
      dtm2 <- TermDocumentMatrix(docs)
      m2 <- as.matrix(dtm2)
      v2 <- sort(rowSums(m2),decreasing=TRUE)
      d2 <- data.frame(word = names(v2),freq=v2)
      wordcloud(words = d2$word, freq = d2$freq, min.freq = input$minfreqw, 
                max.words=input$maxnumw, random.order=FALSE, rot.per=0, 
                colors=brewer.pal(8, "Dark2"))
    })
    
  })
  
  



  })

}

# Run the application 
shinyApp(ui = ui, server = server)
