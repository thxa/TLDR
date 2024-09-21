let autoRefreshIntervalId = null;
let initialized = false;
let optimizing = false;
let demoDataId = null;
let scheduleId = null;
let loadedRoutePlan = null;
let newVisit = null;
let visitMarker = null;
const solveButton = $('#solveButton');
const stopSolvingButton = $('#stopSolvingButton');
const vehiclesTable = $('#vehicles');
const analyzeButton = $('#analyzeButton');

/*************************************** Map constants and variable definitions  **************************************/

const homeLocationMarkerByIdMap = new Map();
const visitMarkerByIdMap = new Map();

const map = L.map('map', {doubleClickZoom: false}).setView([51.505, -0.09], 13);
const visitGroup = L.layerGroup().addTo(map);
const homeLocationGroup = L.layerGroup().addTo(map);
const routeGroup = L.layerGroup().addTo(map);

/************************************ Time line constants and variable definitions ************************************/

const byVehiclePanel = document.getElementById("byVehiclePanel");
const byVehicleTimelineOptions = {
    timeAxis: {scale: "hour"},
    orientation: {axis: "top"},
    xss: {disabled: true}, // Items are XSS safe through JQuery
    stack: false,
    stackSubgroups: false,
    zoomMin: 1000 * 60 * 60, // A single hour in milliseconds
    zoomMax: 1000 * 60 * 60 * 24 // A single day in milliseconds
};
const byVehicleGroupData = new vis.DataSet();
const byVehicleItemData = new vis.DataSet();
const byVehicleTimeline = new vis.Timeline(byVehiclePanel, byVehicleItemData, byVehicleGroupData, byVehicleTimelineOptions);

const byVisitPanel = document.getElementById("byVisitPanel");
const byVisitTimelineOptions = {
    timeAxis: {scale: "hour"},
    orientation: {axis: "top"},
    verticalScroll: true,
    xss: {disabled: true}, // Items are XSS safe through JQuery
    stack: false,
    stackSubgroups: false,
    zoomMin: 1000 * 60 * 60, // A single hour in milliseconds
    zoomMax: 1000 * 60 * 60 * 24 // A single day in milliseconds
};
const byVisitGroupData = new vis.DataSet();
const byVisitItemData = new vis.DataSet();
const byVisitTimeline = new vis.Timeline(byVisitPanel, byVisitItemData, byVisitGroupData, byVisitTimelineOptions);

/************************************ Initialize ************************************/

$(document).ready(function () {
    replaceQuickstartTimefoldAutoHeaderFooter();

    // L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    //     maxZoom: 19,
    //     attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors',
    // }).addTo(map);
    
    map.attributionControl.setPrefix('');

    // Add OpenStreetMap layer
    var osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors',
    }).addTo(map);

    // Add Google Maps layer
    var googleLayer = L.tileLayer('https://mt{s}.google.com/vt?x={x}&y={y}&z={z}&s=Ga', {
        subdomains: '0123',
        maxZoom: 20,
        attribution: '&copy; <a href="https://www.google.com/maps">Google</a>',
    });

    // Add Esri layer
    var esriLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.esri.com/">Esri</a>',
    });

    // Base layer control
    var baseLayers = {
        "OpenStreetMap": osmLayer,
        "Google Maps": googleLayer,
        "Esri": esriLayer
    };

    // Add layer control to switch between map layers
    L.control.layers(baseLayers).addTo(map);


    solveButton.click(solve);
    stopSolvingButton.click(stopSolving);
    analyzeButton.click(analyze);
    refreshSolvingButtons(false);

    // HACK to allow vis-timeline to work within Bootstrap tabs
    $("#byVehicleTab").on('shown.bs.tab', function (event) {
        byVehicleTimeline.redraw();
    })
    $("#byVisitTab").on('shown.bs.tab', function (event) {
        byVisitTimeline.redraw();
    })
    // Add new visit
    map.on('click', function (e) {
        visitMarker = L.circleMarker(e.latlng);
        visitMarker.setStyle({color: 'green'});
        visitMarker.addTo(map);
        openRecommendationModal(e.latlng.lat, e.latlng.lng);
    });
    // Remove visit mark
    $("#newVisitModal").on("hidden.bs.modal", function () {
        map.removeLayer(visitMarker);
    });
    setupAjax();
    fetchDemoData();
});

function colorByVehicle(vehicle) {
    return vehicle === null ? null : pickColor('vehicle' + vehicle.id);
}

function formatDrivingTime(drivingTimeInSeconds) {
    return `${Math.floor(drivingTimeInSeconds / 3600)}h ${Math.round((drivingTimeInSeconds % 3600) / 60)}m`;
}

function homeLocationPopupContent(vehicle) {
    return `<h5>Vehicle ${vehicle.id}</h5>
Home Location`;
}

function visitPopupContent(visit) {
    const arrival = visit.arrivalTime ? `<h6>Arrival at ${showTimeOnly(visit.arrivalTime)}.</h6>` : '';
    return `<h5>${visit.name}</h5>
    <h6>Demand: ${visit.demand}</h6>
    <h6>Available from ${showTimeOnly(visit.minStartTime)} to ${showTimeOnly(visit.maxEndTime)}.</h6>
    ${arrival}`;
}

function showTimeOnly(localDateTimeString) {
    return JSJoda.LocalDateTime.parse(localDateTimeString).toLocalTime();
}

function getHomeLocationMarker(vehicle) {
    let marker = homeLocationMarkerByIdMap.get(vehicle.id);
    if (marker) {
        return marker;
    }
    marker = L.circleMarker(vehicle.homeLocation, { color: colorByVehicle(vehicle), fillOpacity: 0.8 });
    marker.addTo(homeLocationGroup).bindPopup();
    homeLocationMarkerByIdMap.set(vehicle.id, marker);
    return marker;
}

function getVisitMarker(visit) {
    let marker = visitMarkerByIdMap.get(visit.id);
    if (marker) {
        return marker;
    }
    marker = L.circleMarker(visit.location);
    marker.addTo(visitGroup).bindPopup();
    visitMarkerByIdMap.set(visit.id, marker);
    return marker;
}

function renderRoutes(solution) {
    if (!initialized) {
        const bounds = [solution.southWestCorner, solution.northEastCorner];
        map.fitBounds(bounds);
    }
    // Vehicles
    vehiclesTable.children().remove();
    solution.vehicles.forEach(function (vehicle) {
        getHomeLocationMarker(vehicle).setPopupContent(homeLocationPopupContent(vehicle));
        const {id, capacity, totalDemand, totalDrivingTimeSeconds} = vehicle;
        const percentage = totalDemand / capacity * 100;
        const color = colorByVehicle(vehicle);
        vehiclesTable.append(`
      <tr>
        <td>
          <i class="fas fa-crosshairs" id="crosshairs-${id}"
            style="background-color: ${color}; display: inline-block; width: 1rem; height: 1rem; text-align: center">
          </i>
        </td>
        <td>Vehicle ${id}</td>
        <td>
          <div class="progress" data-bs-toggle="tooltip-load" data-bs-placement="left" data-html="true"
            title="Cargo: ${totalDemand} / Capacity: ${capacity}">
            <div class="progress-bar" role="progressbar" style="width: ${percentage}%">${totalDemand}/${capacity}</div>
          </div>
        </td>
        <td>${formatDrivingTime(totalDrivingTimeSeconds)}</td>
      </tr>`);
    });
    // Visits
    solution.visits.forEach(function (visit) {
        getVisitMarker(visit).setPopupContent(visitPopupContent(visit));
    });
    // Route
    routeGroup.clearLayers();
    const visitByIdMap = new Map(solution.visits.map(visit => [visit.id, visit]));
    for (let vehicle of solution.vehicles) {
        const homeLocation = vehicle.homeLocation;
        const locations = vehicle.visits.map(visitId => visitByIdMap.get(visitId).location);
        L.polyline([homeLocation, ...locations, homeLocation], {color: colorByVehicle(vehicle)}).addTo(routeGroup);
    }

    // Summary
    $('#score').text(solution.score);
    $('#drivingTime').text(formatDrivingTime(solution.totalDrivingTimeSeconds));
}

function renderTimelines(routePlan) {
    byVehicleGroupData.clear();
    byVisitGroupData.clear();
    byVehicleItemData.clear();
    byVisitItemData.clear();

    $.each(routePlan.vehicles, function (index, vehicle) {
        const {totalDemand, capacity} = vehicle
        const percentage = totalDemand / capacity * 100;
        const vehicleWithLoad = `<h5 class="card-title mb-1">vehicle-${vehicle.id}</h5>
                                 <div class="progress" data-bs-toggle="tooltip-load" data-bs-placement="left" 
                                      data-html="true" title="Cargo: ${totalDemand} / Capacity: ${capacity}">
                                   <div class="progress-bar" role="progressbar" style="width: ${percentage}%">
                                      ${totalDemand}/${capacity}
                                   </div>
                                 </div>`
        byVehicleGroupData.add({id: vehicle.id, content: vehicleWithLoad});
    });

    $.each(routePlan.visits, function (index, visit) {
        const minStartTime = JSJoda.LocalDateTime.parse(visit.minStartTime);
        const maxEndTime = JSJoda.LocalDateTime.parse(visit.maxEndTime);
        const serviceDuration = JSJoda.Duration.ofSeconds(visit.serviceDuration);

        const visitGroupElement = $(`<div/>`)
            .append($(`<h5 class="card-title mb-1"/>`).text(`${visit.name}`));
        byVisitGroupData.add({
            id: visit.id,
            content: visitGroupElement.html()
        });

        // Time window per visit.
        byVisitItemData.add({
            id: visit.id + "_readyToDue",
            group: visit.id,
            start: visit.minStartTime,
            end: visit.maxEndTime,
            type: "background",
            style: "background-color: #8AE23433"
        });

        if (visit.vehicle == null) {
            const byJobJobElement = $(`<div/>`)
                .append($(`<h5 class="card-title mb-1"/>`).text(`Unassigned`));

            // Unassigned are shown at the beginning of the visit's time window; the length is the service duration.
            byVisitItemData.add({
                id: visit.id + '_unassigned',
                group: visit.id,
                content: byJobJobElement.html(),
                start: minStartTime.toString(),
                end: minStartTime.plus(serviceDuration).toString(),
                style: "background-color: #EF292999"
            });
        } else {
            const arrivalTime = JSJoda.LocalDateTime.parse(visit.arrivalTime);
            const beforeReady = arrivalTime.isBefore(minStartTime);
            const arrivalPlusService = arrivalTime.plus(serviceDuration);
            const afterDue = arrivalPlusService.isAfter(maxEndTime);

            const byVehicleElement = $(`<div/>`)
                .append('<div/>')
                .append($(`<h5 class="card-title mb-1"/>`).text(visit.name));

            const byVisitElement = $(`<div/>`)
                // visit.vehicle is the vehicle.id due to Jackson serialization
                .append($(`<h5 class="card-title mb-1"/>`).text('vehicle-' + visit.vehicle));

            const byVehicleTravelElement = $(`<div/>`)
                .append($(`<h5 class="card-title mb-1"/>`).text('Travel'));

            const previousDeparture = arrivalTime.minusSeconds(visit.drivingTimeSecondsFromPreviousStandstill);
            byVehicleItemData.add({
                id: visit.id + '_travel',
                group: visit.vehicle, // visit.vehicle is the vehicle.id due to Jackson serialization
                subgroup: visit.vehicle,
                content: byVehicleTravelElement.html(),
                start: previousDeparture.toString(),
                end: visit.arrivalTime,
                style: "background-color: #f7dd8f90"
            });
            if (beforeReady) {
                const byVehicleWaitElement = $(`<div/>`)
                    .append($(`<h5 class="card-title mb-1"/>`).text('Wait'));

                byVehicleItemData.add({
                    id: visit.id + '_wait',
                    group: visit.vehicle, // visit.vehicle is the vehicle.id due to Jackson serialization
                    subgroup: visit.vehicle,
                    content: byVehicleWaitElement.html(),
                    start: visit.arrivalTime,
                    end: visit.minStartTime
                });
            }
            let serviceElementBackground = afterDue ? '#EF292999' : '#83C15955'

            byVehicleItemData.add({
                id: visit.id + '_service',
                group: visit.vehicle, // visit.vehicle is the vehicle.id due to Jackson serialization
                subgroup: visit.vehicle,
                content: byVehicleElement.html(),
                start: visit.startServiceTime,
                end: visit.departureTime,
                style: "background-color: " + serviceElementBackground
            });
            byVisitItemData.add({
                id: visit.id,
                group: visit.id,
                content: byVisitElement.html(),
                start: visit.startServiceTime,
                end: visit.departureTime,
                style: "background-color: " + serviceElementBackground
            });

        }

    });

    $.each(routePlan.vehicles, function (index, vehicle) {
        if (vehicle.visits.length > 0) {
            let lastVisit = routePlan.visits.filter((visit) => visit.id == vehicle.visits[vehicle.visits.length -1]).pop();
            if (lastVisit) {
                byVehicleItemData.add({
                    id: vehicle.id + '_travelBackToHomeLocation',
                    group: vehicle.id, // visit.vehicle is the vehicle.id due to Jackson serialization
                    subgroup: vehicle.id,
                    content: $(`<div/>`).append($(`<h5 class="card-title mb-1"/>`).text('Travel')).html(),
                    start: lastVisit.departureTime,
                    end: vehicle.arrivalTime,
                    style: "background-color: #f7dd8f90"
                });
            }
        }
    });

    if (!initialized) {
        byVehicleTimeline.setWindow(routePlan.startDateTime, routePlan.endDateTime);
        byVisitTimeline.setWindow(routePlan.startDateTime, routePlan.endDateTime);
    }
}

function analyze() {
    // see score-analysis.js
    analyzeScore(loadedRoutePlan, "/route-plans/analyze")
}

// TODO: move the general functionality to the webjar.

function setupAjax() {
    $.ajaxSetup({
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json,text/plain', // plain text is required by solve() returning UUID of the solver job
        }
    });

    // Extend jQuery to support $.put() and $.delete()
    jQuery.each(["put", "delete"], function (i, method) {
        jQuery[method] = function (url, data, callback, type) {
            if (jQuery.isFunction(data)) {
                type = type || callback;
                callback = data;
                data = undefined;
            }
            return jQuery.ajax({
                url: url,
                type: method,
                dataType: type,
                data: data,
                success: callback
            });
        };
    });
}

function solve() {
    $.post("/route-plans", JSON.stringify(loadedRoutePlan), function (data) {
        scheduleId = data;
        refreshSolvingButtons(true);
    }).fail(function (xhr, ajaxOptions, thrownError) {
            showError("Start solving failed.", xhr);
            refreshSolvingButtons(false);
        },
        "text");
}

function refreshSolvingButtons(solving) {
    optimizing = solving;
    if (solving) {
        $("#solveButton").hide();
        $("#visitButton").hide();
        $("#stopSolvingButton").show();
        if (autoRefreshIntervalId == null) {
            autoRefreshIntervalId = setInterval(refreshRoutePlan, 2000);
        }
    } else {
        $("#solveButton").show();
        $("#visitButton").show();
        $("#stopSolvingButton").hide();
        if (autoRefreshIntervalId != null) {
            clearInterval(autoRefreshIntervalId);
            autoRefreshIntervalId = null;
        }
    }
}

function refreshRoutePlan() {
    let path = "/route-plans/" + scheduleId;
    if (scheduleId === null) {
        if (demoDataId === null) {
            alert("Please select a test data set.");
            return;
        }

        path = "/demo-data/" + demoDataId;
    }

    $.getJSON(path, function (routePlan) {
        loadedRoutePlan = routePlan;
        refreshSolvingButtons(routePlan.solverStatus != null && routePlan.solverStatus !== "NOT_SOLVING");
        renderRoutes(routePlan);
        renderTimelines(routePlan);
        initialized = true;
    }).fail(function (xhr, ajaxOptions, thrownError) {
        showError("Getting route plan has failed.", xhr);
        refreshSolvingButtons(false);
    });
}

function stopSolving() {
    $.delete("/route-plans/" + scheduleId, function () {
        refreshSolvingButtons(false);
        refreshRoutePlan();
    }).fail(function (xhr, ajaxOptions, thrownError) {
        showError("Stop solving failed.", xhr);
    });
}

function fetchDemoData() {
    $.get("/demo-data", function (data) {
        data.forEach(function (item) {
            $("#testDataButton").append($('<a id="' + item + 'TestData" class="dropdown-item" href="#">' + item + '</a>'));

            $("#" + item + "TestData").click(function () {
                switchDataDropDownItemActive(item);
                scheduleId = null;
                demoDataId = item;
                initialized = false;
                homeLocationGroup.clearLayers();
                homeLocationMarkerByIdMap.clear();
                visitGroup.clearLayers();
                visitMarkerByIdMap.clear();
                refreshRoutePlan();
            });
        });

        demoDataId = data[0];
        switchDataDropDownItemActive(demoDataId);

        refreshRoutePlan();
    }).fail(function (xhr, ajaxOptions, thrownError) {
        // disable this page as there is no data
        $("#demo").empty();
        $("#demo").html("<h1><p style=\"justify-content: center\">No test data available</p></h1>")
    });
}

function switchDataDropDownItemActive(newItem) {
    activeCssClass = "active";
    $("#testDataButton > a." + activeCssClass).removeClass(activeCssClass);
    $("#" + newItem + "TestData").addClass(activeCssClass);
}

function copyTextToClipboard(id) {
    var text = $("#" + id).text().trim();

    var dummy = document.createElement("textarea");
    document.body.appendChild(dummy);
    dummy.value = text;
    dummy.select();
    document.execCommand("copy");
    document.body.removeChild(dummy);
}

function replaceQuickstartTimefoldAutoHeaderFooter() {
    const timefoldHeader = $("header#timefold-auto-header");
    if (timefoldHeader != null) {
        timefoldHeader.addClass("bg-black")
        timefoldHeader.append(
            $(`<div class="container-fluid">
        <nav class="navbar sticky-top navbar-expand-lg navbar-dark shadow mb-3">
          <a class="navbar-brand" href="">
            <img src="/webjars/timefold/img/logo.png" alt="EcoRoute" width="100" height="50">
          </a>
          <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
          </button>
          <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="nav nav-pills">
              <li class="nav-item active" id="navUIItem">
                <button class="nav-link active" id="navUI" data-bs-toggle="pill" data-bs-target="#demo" type="button">Demo UI</button>
              </li>
              <li class="nav-item" id="navOpenApiItem">
                <button class="nav-link" id="navOpenApi" data-bs-toggle="pill" data-bs-target="#openapi" type="button">REST API</button>
              </li>
            </ul>
          </div>
          <div class="ms-auto">
              <div class="btn-group dropstart">
                  <div id="testDataButton" class="dropdown-menu" aria-labelledby="dropdownMenuButton"></div>
              </div>
          </div>
        </nav>
      </div>`));
    }

    const timefoldFooter = $("footer#timefold-auto-footer");
    if (timefoldFooter != null) {
        timefoldFooter.append(
            $(`<footer class="bg-black text-white-50">
               <div class="container">
                 <div class="hstack gap-3 p-4">
                   <div class="ms-auto"><a class="text-white" href=""></a></div>
                   <div class="vr"></div>
                   <div class="me-auto"><a class="text-white" href=""></a></div>
                 </div>
               </div>
             </footer>`));
    }
}
