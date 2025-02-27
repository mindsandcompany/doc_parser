const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3003;

// 폴더 경로 설정
const FOLDER_TO_SERVE = path.join(__dirname, '../../output2');
const FOLDER_TO_SERVE_JSON = path.join(__dirname, '../../output2');

// 정적 파일 제공
app.use(express.static('public'));

// 이미지 파일 확장자 목록
const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'];
let fileindex = 1
// 폴더 트리 가져오기 (이미지 파일만, "artifacts" 폴더 제외)
function getImageTree(dir) {
    const items = fs.readdirSync(dir, { withFileTypes: true });
    
    const collator = new Intl.Collator(undefined, {numeric:true, sensitivity:'base'})
    items.sort((a,b) => collator.compare(a.name, b.name))
    
    return items
        .map(item => {
            const fullPath = path.join(dir, item.name);

            // "artifacts" 폴더 제외
            if (item.isDirectory() && item.name === 'artifacts') {
                return null;
            }

            // if (item.isDirectory() && item.name === '지침서') {
            //     return null;
            // }

            if (item.isDirectory()) {
                return {
                    name: item.name,
                    type: 'folder',
                    children: getImageTree(fullPath) // 하위 폴더 탐색
                };
            } else if (item.isFile() && IMAGE_EXTENSIONS.includes(path.extname(item.name).toLowerCase())) {
                return {
                    name: item.name,
                    name_dsp: item.name.slice(0,-4).match(/\d{1,3}$/)[0],
                    fileindex: fileindex++,
                    type: 'file',
                    path: fullPath.replace(FOLDER_TO_SERVE, '') // 상대 경로
                };
            }

            return null; // 다른 항목 제외
        })
        .filter(Boolean); // null 값 제거
}

// 특정 폴더의 docling_info.json 읽기
function getChunkInfo_hybrid(folderPath) {
    const jsonPath = path.join(folderPath, 'result_chunks_hybrid.json');
    if (fs.existsSync(jsonPath)) {
        try {
            const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
            return data;
        } catch (err) {
            console.error(`Error parsing JSON at ${jsonPath}:`, err);
            return null;
        }
    }
    return null;
}
function getChunkInfo(folderPath) {
    const jsonPath = path.join(folderPath, 'result_chunks.json');
    if (fs.existsSync(jsonPath)) {
        try {
            const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
            return data;
        } catch (err) {
            console.error(`Error parsing JSON at ${jsonPath}:`, err);
            return null;
        }
    }
    return null;
}
function getDoclingInfo(folderPath) {
    const jsonPath = path.join(folderPath, 'result_edit.json');
    if (fs.existsSync(jsonPath)) {
        try {
            const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
            return data;
        } catch (err) {
            console.error(`Error parsing JSON at ${jsonPath}:`, err);
            return null;
        }
    }
    return null;
}
function getDoclingInfo_origin(folderPath) {
    const jsonPath = path.join(folderPath, 'result.json');
    if (fs.existsSync(jsonPath)) {
        try {
            const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
            return data;
        } catch (err) {
            console.error(`Error parsing JSON at ${jsonPath}:`, err);
            return null;
        }
    }
    return null;
}

// docling_info.json API
app.get('/api/chunks', (req, res) => {
    const folderPath = path.join(FOLDER_TO_SERVE_JSON, req.query.folder || '');
    const info = getChunkInfo(folderPath);
    if (info) {
        res.json(info);
    } else {
        res.status(404).send('docling_info.json not found');
    }
});
app.get('/api/chunks_hybrid', (req, res) => {
    const folderPath = path.join(FOLDER_TO_SERVE_JSON, req.query.folder || '');
    const info = getChunkInfo_hybrid(folderPath);
    if (info) {
        res.json(info);
    } else {
        res.status(404).send('docling_info.json not found');
    }
});
app.get('/api/docling', (req, res) => {
    const folderPath = path.join(FOLDER_TO_SERVE_JSON, req.query.folder || '');
    const info = getDoclingInfo(folderPath);
    if (info) {
        res.json(info);
    } else {
        res.status(404).send('docling_info.json not found');
    }
});
app.get('/api/docling_origin', (req, res) => {
    const folderPath = path.join(FOLDER_TO_SERVE_JSON, req.query.folder || '');
    const info = getDoclingInfo_origin(folderPath);
    if (info) {
        res.json(info);
    } else {
        res.status(404).send('docling_info.json not found');
    }
});
// 폴더 트리 API
app.get('/api/tree', (req, res) => {
    fileindex = 1
    const tree = getImageTree(FOLDER_TO_SERVE);
    console.log("last fileindex:", fileindex)
    res.json(tree);
});

// 이미지 파일 서빙
app.get('/file/*', (req, res) => {
    const filePath = path.join(FOLDER_TO_SERVE, req.params[0]);
    //console.log("filePath:", filePath)
    if (IMAGE_EXTENSIONS.includes(path.extname(filePath).toLowerCase())) {
        res.sendFile(decodeURIComponent(filePath));
    } else {
        //console.log("filePath:", filePath)
        res.status(404).send('File not found');
    }
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
