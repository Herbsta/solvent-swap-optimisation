
'use client';

import { Button } from "@/components/ui/button";

const DBDownloadButton = () => {
    const handleDownload = async () => {
        try {
            const response = await fetch('/download/api')

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.message || 'Download failed')
            }

            const blob = await response.blob()
            const url = window.URL.createObjectURL(blob)

            const link = document.createElement('a')
            link.href = url
            link.download = 'database.db'

            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)

            window.URL.revokeObjectURL(url)
        } catch (error) {
            console.error('Download error:', error)
            alert('Failed to download database: ' + (error as Error).message)
        }
    }
    return (
        <Button onClick={() => handleDownload()}>
            Download Database
        </Button>
    )
}

export default DBDownloadButton