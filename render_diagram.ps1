Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$url = "file:///c:/Users/777Pr/Downloads/Research%20implementation/render.html"
$outPath = "c:\Users\777Pr\Downloads\Research implementation\fig1_pipeline_diagram.png"
$outPlotsPath = "c:\Users\777Pr\Downloads\Research implementation\output\part2\plots\fig1_pipeline_diagram.png"

$browser = New-Object System.Windows.Forms.WebBrowser
$browser.ScrollBarsEnabled = $false
$browser.Width = 800
$browser.Height = 1260
$browser.Navigate($url)

$timeout = 0
while ($browser.ReadyState -ne [System.Windows.Forms.WebBrowserReadyState]::Complete -and $timeout -lt 50) {
    [System.Windows.Forms.Application]::DoEvents()
    Start-Sleep -Milliseconds 100
    $timeout++
}

Start-Sleep -Milliseconds 500
[System.Windows.Forms.Application]::DoEvents()

$bitmap = New-Object System.Drawing.Bitmap(800, 1260)
$browser.DrawToBitmap($bitmap, (New-Object System.Drawing.Rectangle(0, 0, 800, 1260)))
$bitmap.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)
$bitmap.Save($outPlotsPath, [System.Drawing.Imaging.ImageFormat]::Png)
Write-Output "Saved PNG to $outPath and $outPlotsPath"
