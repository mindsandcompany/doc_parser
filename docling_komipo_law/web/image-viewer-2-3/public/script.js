// 전역 상태 변수
let showTextBoxes = true;
let showGroupBoxes = true;
let showChunkBoxes = true;
let showChunkBoxes_hybrid = true;
let showOriginJson = false;

let currentImagePath = null;
let currentDoclingInfo = null;
let currentPageNumber = null;
let currentChunkInfo = null;
let currentChunkInfo_hybrid = null;

let doclingInfo_ = null
let chunkInfo_ = null
let chunkInfo_hybrid_ = null
let folderPathForJson_ = null
let imagePath_ = null
let encodedImagePath = null
let pageNumber_ = null
let currentFileIndex_ = 0

const breadcrumb = document.getElementById('file-path');

// 체크박스 이벤트 핸들러
document.getElementById('toggle-text').addEventListener('change', (e) => {
    showTextBoxes = e.target.checked;
    redraw(); // 박스 다시 그리기
});

document.getElementById('toggle-group').addEventListener('change', (e) => {
    showGroupBoxes = e.target.checked;
    redraw(); // 박스 다시 그리기
});

document.getElementById('toggle-chunk').addEventListener('change', (e) => {
    showChunkBoxes = e.target.checked;
    redraw(); // 박스 다시 그리기
});
document.getElementById('toggle-chunk-hybrid').addEventListener('change', (e) => {
    showChunkBoxes_hybrid = e.target.checked;
    redraw(); // 박스 다시 그리기
});
document.getElementById('toggle-origin').addEventListener('change', async (e) => {
    showOriginJson = e.target.checked;
    if(showOriginJson == true){
        doclingInfo_ = await fetchDoclingInfo_origin(folderPathForJson_);
    } else {
        doclingInfo_ = await fetchDoclingInfo(folderPathForJson_);
    }
    drawImageWithBoxes(encodedImagePath, doclingInfo_, pageNumber_, chunkInfo_, chunkInfo_hybrid_);
    //redraw(); // 박스 다시 그리기
});
// document.getElementById('viewjson').addEventListener('click', async (e) => {
//     const doclingInfo = await fetchDoclingInfo_origin(folderPathForJson_);
//     drawImageWithBoxes(encodedImagePath, doclingInfo, pageNumber_);
// });

// document.getElementById('viewjson_edit').addEventListener('click', (e) => {
//     drawImageWithBoxes(encodedImagePath, doclingInfo_, pageNumber_);
// });


document.getElementById('showJsonOrigin').addEventListener('click', async (e) => {
    const doclingInfo__ = await fetchDoclingInfo_origin(folderPathForJson_);
    const jsonString = JSON.stringify(doclingInfo__, null, 2)
    const popup = window.open('', 'json원본', 'width=600, height=400');
    popup.document.write('<pre>' + jsonString + '</pre>')
});
document.getElementById('showJsonButton').addEventListener('click', (e) => {
    const jsonString = JSON.stringify(doclingInfo_, null, 2)
    const popup = window.open('', 'json편집본', 'width=600, height=400');
    popup.document.write('<pre>' + jsonString + '</pre>')
});
document.getElementById('showChunkButton').addEventListener('click', async (e) => {
    const doclingInfo__ = await fetchChunkInfo(folderPathForJson_);
    const jsonString = JSON.stringify(doclingInfo__, null, 2)
    const popup = window.open('', 'json_Chunk', 'width=600, height=400');
    popup.document.write('<pre>' + jsonString + '</pre>')
});
document.getElementById('showChunkHybridButton').addEventListener('click', async (e) => {
    const doclingInfo__ = await fetchChunkInfo_hybrid(folderPathForJson_);
    const jsonString = JSON.stringify(doclingInfo__, null, 2)
    const popup = window.open('', 'json_Chunk_hybrid', 'width=600, height=400');
    popup.document.write('<pre>' + jsonString + '</pre>')
});

function setParentBlock(element){
    let currentElement = element
    while (currentElement && currentElement instanceof HTMLElement) {
        const computedStyle = window.getComputedStyle(currentElement)
        if (computedStyle.display === 'none')
        {
            currentElement.style.display = 'block'
        }
        currentElement = currentElement.parentNode
    }
}
function setParentNone(element){
    let currentElement = element
    while (currentElement && currentElement instanceof HTMLElement) {
        const computedStyle = window.getComputedStyle(currentElement)
        if (computedStyle.display === 'block')
        {
            currentElement.style.display = 'none'
        }
        currentElement = currentElement.parentNode
    }
}

document.getElementById('file-path').addEventListener('click', (e) => {
    const input = document.createElement("input");
    input.type = "text"
    input.value = imagePath_
    input.style.width = "100%"

    breadcrumb.replaceWith(input)

    input.focus()

    input.addEventListener("keydown", async (e)=>{
        if (e.key === "Enter"){
            const newFilePath = input.value.replace(/^\/file/,'')
            // console.log("newFilePath:", newFilePath)
            const treeContainer = document.querySelector('#tree-view')
            const selectedDiv = treeContainer.querySelector('.selected')
            const nextDiv = treeContainer.querySelector(`div[filepath="${newFilePath}"]`)
            currentFileIndex_ = parseInt(nextDiv.getAttribute('fileindex'), 10)
            
            const nextFolderPath = nextDiv.getAttribute('folderpath')

            if (selectedDiv)
            {
                selectedDiv.parentNode.style.display = 'none';
                selectedDiv.classList.remove('selected')
                setParentNone(selectedDiv)
                
            }

            // const currentFolderPath = selectedDiv.getAttribute('folderpath')
            // if (!currentFolderPath || currentFolderPath != nextFolderPath){
            //     console.log("Folder has changed!")
            //     nextDiv.parentNode.style.display = 'block';
            // }
            
            if (nextDiv)
            {
                nextDiv.parentNode.style.display = 'block';
                setParentBlock(nextDiv)
                nextDiv.classList.add('selected')
                nextDiv.scrollIntoView({behavior:'smooth', block: 'center', inline:'nearest'})
                const nextdiv_filepath = nextDiv.getAttribute('filepath')
                const nextdiv_filename = nextDiv.getAttribute('filename')
                const nextdiv_folderpath = nextDiv.getAttribute('folderpath')

                const encodedPath = nextdiv_filepath.split('/').map(encodeURIComponent).join('/')
                const imagePath = `/file${encodedPath}`;
                const folderPathForJson = nextdiv_folderpath;
                folderPathForJson_ = folderPathForJson
                imagePath_ = nextdiv_filepath
                encodedImagePath = imagePath

                // docling_info.json 데이터 가져오기
                if(showOriginJson == true){
                    doclingInfo_ = await fetchDoclingInfo_origin(folderPathForJson);
                } else {
                    doclingInfo_ = await fetchDoclingInfo(folderPathForJson);
                }
                //doclingInfo_ = doclingInfo
                const chunkInfo = await fetchChunkInfo(folderPathForJson);
                chunkInfo_ = chunkInfo
                const chunkInfo_hybrid = await fetchChunkInfo_hybrid(folderPathForJson);
                chunkInfo_hybrid_ = chunkInfo_hybrid
                // 파일명에서 페이지 번호 추출
                const match = nextdiv_filename.match(/(\d+)\.png$/);
                const pageNumber = parseInt(match[1], 10);
                // console.log(pageNumber)
                pageNumber_ = pageNumber

                if (pageNumber !== null) {
                    drawImageWithBoxes(imagePath, doclingInfo_, pageNumber, chunkInfo, chunkInfo_hybrid);
                    breadcrumb.innerHTML = nextdiv_filepath
                    if (input.isConnected)
                        input.replaceWith(breadcrumb)
                } else {
                    console.error(`Invalid file format: ${nextdiv_filename}`);
                }

                
            }
        } else if (e.key === "Escape"){
            if (input.isConnected)
                input.replaceWith(breadcrumb)
        }
    })

    input.addEventListener("blur", ()=>{
        if (input.isConnected)
            input.replaceWith(breadcrumb)
    })
})


document.addEventListener('keydown', async (event) => {
    if (event.key === 'ArrowRight')
    {
        const activeElement = document.activeElement
        if (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA'){
            return
        }
        const treeContainer = document.querySelector('#tree-view')
        const selectedDiv = treeContainer.querySelector('.selected')
        if (selectedDiv)
        {
            //console.log("selectedDiv not found")
            //selectedDiv.parentNode.style.display = 'none';
            selectedDiv.classList.remove('selected')
            // return
        }
        const currentFileIndex = parseInt(selectedDiv.getAttribute('fileindex'), 10)
        const nextDiv = treeContainer.querySelector(`div[fileindex="${currentFileIndex+1}"]`)
        currentFileIndex_ = parseInt(nextDiv.getAttribute('fileindex'), 10)

        const currentFolderPath = selectedDiv.getAttribute('folderpath')
        const nextFolderPath = nextDiv.getAttribute('folderpath')
        if (!currentFolderPath || currentFolderPath != nextFolderPath){
            
            selectedDiv.parentNode.style.display = 'none';
            nextDiv.parentNode.style.display = 'block';
        }

        if (nextDiv)
        {
            nextDiv.classList.add('selected')
            nextDiv.scrollIntoView({behavior:'smooth', block: 'center', inline:'nearest'})
            const nextdiv_filepath = nextDiv.getAttribute('filepath')
            const nextdiv_filename = nextDiv.getAttribute('filename')
            const nextdiv_folderpath = nextDiv.getAttribute('folderpath')

            const encodedPath = nextdiv_filepath.split('/').map(encodeURIComponent).join('/')
            const imagePath = `/file${encodedPath}`;
            const folderPathForJson = nextdiv_folderpath;
            folderPathForJson_ = folderPathForJson
            imagePath_ = nextdiv_filepath
            encodedImagePath = imagePath

            // docling_info.json 데이터 가져오기
            if(showOriginJson == true){
                doclingInfo_ = await fetchDoclingInfo_origin(folderPathForJson);
            } else {
                doclingInfo_ = await fetchDoclingInfo(folderPathForJson);
            }
            //doclingInfo_ = doclingInfo
            const chunkInfo = await fetchChunkInfo(folderPathForJson);
            chunkInfo_ = chunkInfo
            const chunkInfo_hybrid = await fetchChunkInfo_hybrid(folderPathForJson);
            chunkInfo_hybrid_ = chunkInfo_hybrid
            // 파일명에서 페이지 번호 추출
            const match = nextdiv_filename.match(/(\d+)\.png$/);
            const pageNumber = parseInt(match[1], 10);
            // console.log(pageNumber)
            pageNumber_ = pageNumber

            if (pageNumber !== null) {
                drawImageWithBoxes(imagePath, doclingInfo_, pageNumber, chunkInfo, chunkInfo_hybrid);
            } else {
                console.error(`Invalid file format: ${nextdiv_filename}`);
            }
        }
    } else if (event.key === 'ArrowLeft') {
        const activeElement = document.activeElement
        if (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA'){
            return
        }
        const treeContainer = document.querySelector('#tree-view')
        const selectedDiv = treeContainer.querySelector('.selected')
        if (selectedDiv)
        {
            
            selectedDiv.classList.remove('selected')
            //console.log("selectedDiv not found")
            //return
        }
        const currentFileIndex = parseInt(selectedDiv.getAttribute('fileindex'), 10)
        const prevDiv = treeContainer.querySelector(`div[fileindex="${currentFileIndex-1}"]`)
        currentFileIndex_ = parseInt(prevDiv.getAttribute('fileindex'), 10)

        const currentFolderPath = selectedDiv.getAttribute('folderpath')
        const prevFolderPath = prevDiv.getAttribute('folderpath')
        if (!currentFolderPath || currentFolderPath != prevFolderPath){
            selectedDiv.parentNode.style.display = 'none';
            prevDiv.parentNode.style.display = 'block';
        }

        if (prevDiv)
        {
            prevDiv.classList.add('selected')
            prevDiv.scrollIntoView({behavior:'smooth', block: 'center', inline:'nearest'})
            const prevdiv_filepath = prevDiv.getAttribute('filepath')
            const prevdiv_filename = prevDiv.getAttribute('filename')
            const prevdiv_folderpath = prevDiv.getAttribute('folderpath')

            const encodedPath = prevdiv_filepath.split('/').map(encodeURIComponent).join('/')
            const imagePath = `/file${encodedPath}`;
            imagePath_ = prevdiv_filepath
            encodedImagePath = imagePath
            const folderPathForJson = prevdiv_folderpath;
            folderPathForJson_ = folderPathForJson

            // docling_info.json 데이터 가져오기
            if(showOriginJson == true){
                doclingInfo_ = await fetchDoclingInfo_origin(folderPathForJson);
            } else {
                doclingInfo_ = await fetchDoclingInfo(folderPathForJson);
            }
            //doclingInfo_ = doclingInfo
            const chunkInfo = await fetchChunkInfo(folderPathForJson);
            chunkInfo_ = chunkInfo
            const chunkInfo_hybrid = await fetchChunkInfo_hybrid(folderPathForJson);
            chunkInfo_hybrid_ = chunkInfo_hybrid

            // 파일명에서 페이지 번호 추출
            const match = prevdiv_filename.match(/(\d+)\.png$/);
            const pageNumber = parseInt(match[1], 10);
            // console.log(pageNumber)
            pageNumber_ = pageNumber

            if (pageNumber !== null) {
                drawImageWithBoxes(imagePath, doclingInfo_, pageNumber, chunkInfo, chunkInfo_hybrid);
            } else {
                console.error(`Invalid file format: ${prevdiv_filename}`);
            }
        }
    }
    
});


// 데이터 요청 함수
async function fetchTree() {
    const response = await fetch('/api/tree');
    return await response.json();
}
async function fetchChunkInfo(folder) {
    const response = await fetch(`/api/chunks?folder=${encodeURIComponent(folder)}&t=${Date.now()}`);
    if (response.ok) {
        return await response.json();
    } else {
        return null;
    }
}
async function fetchChunkInfo_hybrid(folder) {
    const response = await fetch(`/api/chunks_hybrid?folder=${encodeURIComponent(folder)}&t=${Date.now()}`);
    if (response.ok) {
        return await response.json();
    } else {
        return null;
    }
}
async function fetchDoclingInfo(folder) {
    const response = await fetch(`/api/docling?folder=${encodeURIComponent(folder)}&t=${Date.now()}`);
    if (response.ok) {
        return await response.json();
    } else {
        return null;
    }
}
async function fetchDoclingInfo_origin(folder) {
    const response = await fetch(`/api/docling_origin?folder=${encodeURIComponent(folder)}&t=${Date.now()}`);
    if (response.ok) {
        return await response.json();
    } else {
        return null;
    }
}

// 트리 뷰 생성 함수
function createTreeView(tree, container, folderPath = '') {
    last_page_num = 0
    tree.forEach(item => {
        const element = document.createElement('div');
        element.textContent = item.name_dsp;
        element.classList.add(item.type);

        if (item.type === 'folder') {
            // 폴더의 자식 요소 컨테이너
            const childrenContainer = document.createElement('div');
            childrenContainer.classList.add('children');
            childrenContainer.style.display = 'none'; // 기본적으로 접힘 상태

            // 재귀적으로 하위 트리 생성
            createTreeView(item.children, childrenContainer, `${folderPath}/${item.name}`);

            // 폴더 클릭 이벤트
            element.addEventListener('click', (e) => {
                e.stopPropagation(); // 이벤트 버블링 방지
                const isExpanded = childrenContainer.style.display === 'block';
                childrenContainer.style.display = isExpanded ? 'none' : 'block';

                // 폴더 열림/닫힘 상태 시각적 표시
                if (isExpanded) {
                    element.textContent = `+ ${item.name}`;
                } else {
                    element.textContent = `- ${item.name}`;
                }
            });

            // 초기 폴더 이름에 `+` 추가
            element.textContent = `+ ${item.name}`;
            element.style.cursor = 'pointer';

            // 폴더와 자식 요소 컨테이너 추가
            container.appendChild(element);
            container.appendChild(childrenContainer);
        } else if (item.type === 'file') {
            // 파일 클릭 이벤트
            element.setAttribute("fileindex", item.fileindex)
            element.setAttribute("filepath", item.path)
            element.setAttribute("filename", item.name)
            element.setAttribute("folderpath", folderPath)
            

            element.addEventListener('click', async () => {
                // 선택된 상태 업데이트
                document.querySelectorAll('.tree-view .selected').forEach(selectedItem => {
                    selectedItem.classList.remove('selected');
                });
                currentFileIndex_ = parseInt(element.getAttribute('fileindex'), 10)
                element.classList.add('selected'); // 현재 선택된 요소에 클래스 추가
                //element.scrollIntoView({behavior:'smooth', block: 'center'})
                const encodedPath = item.path.split('/').map(encodeURIComponent).join('/')
                const imagePath = `/file${encodedPath}`;
                imagePath_ = item.path
                encodedImagePath = imagePath
                const folderPathForJson = folderPath;
                folderPathForJson_ = folderPathForJson

                // docling_info.json 데이터 가져오기
                if(showOriginJson == true){
                    doclingInfo_ = await fetchDoclingInfo_origin(folderPathForJson);
                } else {
                    doclingInfo_ = await fetchDoclingInfo(folderPathForJson);
                }
                //doclingInfo_ = doclingInfo
                const chunkInfo = await fetchChunkInfo(folderPathForJson);
                chunkInfo_ = chunkInfo
                const chunkInfo_hybrid = await fetchChunkInfo_hybrid(folderPathForJson);
                chunkInfo_hybrid_ = chunkInfo_hybrid
                // 파일명에서 페이지 번호 추출
                const match = item.name.match(/(\d+)\.png$/);
                const pageNumber = parseInt(match[1], 10);
                // console.log(pageNumber, match)
                pageNumber_ = pageNumber
                if (pageNumber !== null) {
                    drawImageWithBoxes(imagePath, doclingInfo_, pageNumber, chunkInfo, chunkInfo_hybrid);
                } else {
                    console.error(`Invalid file format: ${item.name}`);
                }
            });

            

            // 파일 이름 추가
            container.appendChild(element);
        }
    });
}



// 박스 렌더링 함수
function drawImageWithBoxes(imagePath, doclingInfo, pageNumber, chunkInfo, chunkInfo_hybrid) {
    currentImagePath = imagePath;
    currentDoclingInfo = doclingInfo;
    currentPageNumber = pageNumber+1;
    currentChunkInfo = chunkInfo;
    currentChunkInfo_hybrid = chunkInfo_hybrid;

    redraw();
}

function redraw() {
    //if (!currentImagePath || !currentDoclingInfo || currentPageNumber === null) return;

    const canvas = document.getElementById('canvas');
    
    const ctx = canvas.getContext('2d');
    const image = new Image();

    if (currentImagePath){
        breadcrumb.innerHTML = decodeURIComponent(currentImagePath)
    } else {
        breadcrumb.innerHTML = 'no file selected'
    }

    // breadcrumb
    image.onload = () => {
        canvas.width = image.width;
        canvas.height = image.height;

        // 캔버스 초기화 및 이미지 렌더링
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0);

        // const scaleY = canvas.height / image.height;
        const scaleY = 2;
        const scaleX = 2;

        ctx.fillText(`${currentFileIndex_}/--`, canvas.width/2, 10)
        
        // Chunk_hybrid 박스 렌더링
        if (!showOriginJson && showChunkBoxes_hybrid && currentChunkInfo_hybrid) {
            currentChunkInfo_hybrid.forEach((chunk, index) => {
                const chunkBoxes = [];
                chunk.meta.doc_items.forEach(child => {
                    child.prov.forEach(prov => {
                        if (prov.page_no === currentPageNumber) {
                            chunkBoxes.push(prov.bbox);
                        }
                    });
                });

                if (chunkBoxes.length > 0) {
                    const l = Math.min(...chunkBoxes.map(b => b.l));
                    const t = Math.max(...chunkBoxes.map(b => b.t));
                    const r = Math.max(...chunkBoxes.map(b => b.r));
                    const b = Math.min(...chunkBoxes.map(b => b.b));
                
                    // 박스 크기와 위치를 상하좌우로 2px씩 확장
                    const padding = 6;
                    const x = l * scaleX - padding;
                    const y = canvas.height - t * scaleY - padding; // 위쪽으로 확장
                    const width = r - l + 2 * padding; // 좌우로 확장
                    const height = t - b + 2 * padding; // 아래쪽으로 확장
                
                    ctx.strokeStyle = 'violet';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x, y, width * scaleX, height * scaleY);

                    // 그룹 번호 추출
                    // 그룹 번호를 오른쪽에 표시
                    ctx.fillStyle = 'violet';
                    ctx.font = '8px Arial';
                    ctx.fillText(index+1, x + width * scaleX + 10, y + height * scaleY / 2 + 5);
                }
            });
        }

        // Chunk 박스 렌더링
        if (!showOriginJson && showChunkBoxes && currentChunkInfo) {
            currentChunkInfo.forEach((chunk, index) => {
                const chunkBoxes = [];
                chunk.meta.doc_items.forEach(child => {
                    child.prov.forEach(prov => {
                        if (prov.page_no === currentPageNumber) {
                            chunkBoxes.push(prov.bbox);
                        }
                    });
                });

                if (chunkBoxes.length > 0) {
                    const l = Math.min(...chunkBoxes.map(b => b.l));
                    const t = Math.max(...chunkBoxes.map(b => b.t));
                    const r = Math.max(...chunkBoxes.map(b => b.r));
                    const b = Math.min(...chunkBoxes.map(b => b.b));
                
                    // 박스 크기와 위치를 상하좌우로 2px씩 확장
                    const padding = 4;
                    const x = l * scaleX - padding;
                    const y = canvas.height - t * scaleY - padding; // 위쪽으로 확장
                    const width = r - l + 2 * padding; // 좌우로 확장
                    const height = t - b + 2 * padding; // 아래쪽으로 확장
                
                    ctx.strokeStyle = 'red';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x, y, width * scaleX, height * scaleY);

                    // 그룹 번호 추출
                    // 그룹 번호를 오른쪽에 표시
                    ctx.fillStyle = 'red';
                    ctx.font = '8px Arial';
                    ctx.fillText(index+1, x + width * scaleX + 10, y + height * scaleY / 2 + 5);
                }
            });
        }

        // 그룹 박스 렌더링
        if (showGroupBoxes && currentDoclingInfo.groups) {
            currentDoclingInfo.groups.forEach(group => {
                const groupBoxes = [];
                group.children.forEach(child => {
                    const match = child.$ref.match(/\/texts\/(\d+)$/);
                    const match_table = child.$ref.match(/\/tables\/(\d+)$/);
                    const match_picture = child.$ref.match(/\/pictures\/(\d+)$/);
                    if (match) {
                        const textIndex = parseInt(match[1], 10);
                        const textData = currentDoclingInfo.texts[textIndex];

                        if (textData && textData.prov) {
                            textData.prov.forEach(prov => {
                                if (prov.page_no === currentPageNumber) {
                                    groupBoxes.push(prov.bbox);
                                }
                            });
                        }
                    }
                    if (match_table) {
                        const textIndex = parseInt(match_table[1], 10);
                        const textData = currentDoclingInfo.tables[textIndex];

                        if (textData && textData.prov) {
                            textData.prov.forEach(prov => {
                                if (prov.page_no === currentPageNumber) {
                                    groupBoxes.push(prov.bbox);
                                }
                            });
                        }
                    }
                    if (match_picture) {
                        const textIndex = parseInt(match_picture[1], 10);
                        const textData = currentDoclingInfo.pictures[textIndex];

                        if (textData && textData.prov) {
                            textData.prov.forEach(prov => {
                                if (prov.page_no === currentPageNumber) {
                                    groupBoxes.push(prov.bbox);
                                }
                            });
                        }
                    }
                });

                if (groupBoxes.length > 0) {
                    const l = Math.min(...groupBoxes.map(b => b.l));
                    const t = Math.max(...groupBoxes.map(b => b.t));
                    const r = Math.max(...groupBoxes.map(b => b.r));
                    const b = Math.min(...groupBoxes.map(b => b.b));
                
                    // 박스 크기와 위치를 상하좌우로 2px씩 확장
                    const padding = 2;
                    const x = l * scaleX - padding;
                    const y = canvas.height - t * scaleY - padding; // 위쪽으로 확장
                    const width = r - l + 2 * padding; // 좌우로 확장
                    const height = t - b + 2 * padding; // 아래쪽으로 확장
                
                    ctx.strokeStyle = 'green';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x, y, width * scaleX, height * scaleY);

                    // 그룹 번호 추출
                    const match = group.self_ref.match(/\/groups\/(\d+)$/);
                    if (match) {
                        const groupNumber = match[1];

                        // 그룹 번호를 오른쪽에 표시
                        ctx.fillStyle = 'green';
                        ctx.font = '8px Arial';
                        ctx.fillText(groupNumber, x + width + 5, y + height * scaleY / 2);
                    }
                }
            });
        }

        // 텍스트 박스 렌더링
        if (showTextBoxes && currentDoclingInfo.texts) {
            currentDoclingInfo.texts.forEach(text => {
                text.prov.forEach(prov => {
                    if (prov.page_no === currentPageNumber) {
                        const bbox = prov.bbox;
                        const { l, t, r, b } = bbox;

                        const x = l * scaleX;
                        const y = canvas.height - t * scaleY;
                        const width = r - l;
                        const height = t - b;

                        ctx.strokeStyle = 'orange';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, y , width * scaleX, height * scaleY);

                        const match = text.self_ref.match(/\/texts\/(\d+)$/);
                        if (match) {
                            const serialNumber = match[1];
                            ctx.fillStyle = 'blue';
                            ctx.font = '8px Arial';
                            ctx.fillText(serialNumber, x - 15, y + 5);
                        }
                    }
                });
            });
        }

        // 테이블 박스 렌더링
        if (showTextBoxes && currentDoclingInfo.tables) {
            currentDoclingInfo.tables.forEach(table => {
                table.prov.forEach(prov => {
                    if (prov.page_no === currentPageNumber) {
                        const bbox = prov.bbox;
                        const { l, t, r, b } = bbox;

                        const x = l * scaleX;
                        const y = canvas.height - t * scaleY;
                        const width = r - l;
                        const height = t - b;

                        ctx.strokeStyle = 'orange';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, y , width * scaleX, height * scaleY);

                        const match = table.self_ref.match(/\/tables\/(\d+)$/);
                        if (match) {
                            const serialNumber = 'table '+String(match[1]);
                            ctx.fillStyle = 'blue';
                            ctx.font = '8px Arial';
                            ctx.fillText(serialNumber, x - 20, y + 5);
                        }
                    }
                });
            });
        }

        // 이미지 박스 렌더링
        if (showTextBoxes && currentDoclingInfo.pictures) {
            currentDoclingInfo.pictures.forEach(picture => {
                picture.prov.forEach(prov => {
                    if (prov.page_no === currentPageNumber) {
                        const bbox = prov.bbox;
                        const { l, t, r, b } = bbox;

                        const x = l * scaleX;
                        const y = canvas.height - t * scaleY;
                        const width = r - l;
                        const height = t - b;

                        ctx.strokeStyle = 'orange';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, y , width * scaleX, height * scaleY);

                        const match = picture.self_ref.match(/\/pictures\/(\d+)$/);
                        if (match) {
                            const serialNumber = 'picture '+String(match[1]);
                            ctx.fillStyle = 'blue';
                            ctx.font = '8px Arial';
                            ctx.fillText(serialNumber, x - 25, y + 5);
                        }
                    }
                });
            });
        }
    };

    image.src = currentImagePath;
}

// 초기화
(async function init() {
    const tree = await fetchTree();
    const treeContainer = document.getElementById('tree-view');
    const previewContainer = document.getElementById('preview');
    
    createTreeView(tree, treeContainer);
})();
